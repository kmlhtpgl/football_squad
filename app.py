import os
import io
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from sqlalchemy.exc import IntegrityError
from itsdangerous import URLSafeSerializer
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Pillow (for lineup image)
from PIL import Image, ImageDraw

from db import SessionLocal, init_db, Schedule, Match, Signup, User, Payment


# -------------------- config --------------------
APP_SECRET = os.getenv("APP_SECRET", "change-me-in-prod")
INVITE_CODE = os.getenv("INVITE_CODE", "")  # set to require invite at signup

serializer = URLSafeSerializer(APP_SECRET, salt="football")
session_signer = URLSafeSerializer(APP_SECRET, salt="session")


# -------------------- app --------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html"])
)


@app.on_event("startup")
def startup():
    init_db()


# -------------------- helpers --------------------
def request_path(request: Request) -> str:
    return request.url.path + (("?" + request.url.query) if request.url.query else "")

def require_admin_user(request: Request):
    user = get_current_user(request)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    return user

def load_font(size: int):
    # Try common macOS fonts; fallback to default if not found
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def normalize_name(name: str) -> str:
    n = " ".join(name.strip().split())
    if len(n) < 2:
        raise ValueError("Name too short")
    return n


def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def parse_week_start(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fmt_week(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def month_window(yyyymm: str):
    year, month = map(int, yyyymm.split("-"))
    start = date(year, month, 1)
    end = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return start, end


def pounds_to_pence(amount_str: str) -> int:
    d = Decimal(amount_str).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(d * 100)


def pence_to_gbp_str(pence: int) -> str:
    return f"{(Decimal(pence) / 100):.2f}"


def cost_per_player_pence(total_cost_pounds: int, confirmed_count: int) -> int:
    if confirmed_count <= 0:
        return 0
    total_pence = Decimal(total_cost_pounds) * 100
    cpp = (total_pence / Decimal(confirmed_count)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(cpp)


def pin_hash(pin: str) -> str:
    pin = pin.strip()
    if len(pin) < 4:
        raise ValueError("PIN too short")
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), APP_SECRET.encode("utf-8"), 120_000)
    return dk.hex()


def verify_pin(pin: str, stored: str) -> bool:
    try:
        return hmac.compare_digest(pin_hash(pin), stored)
    except Exception:
        return False


def get_current_user(request: Request):
    cookie = request.cookies.get("fb_session")
    if not cookie:
        return None
    try:
        data = session_signer.loads(cookie)
        uid = int(data["uid"])
    except Exception:
        return None

    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == uid).first()
    finally:
        db.close()


def get_default_schedule(db):
    sched = db.query(Schedule).order_by(Schedule.id.asc()).first()
    if not sched:
        sched = Schedule()
        db.add(sched)
        db.commit()
        db.refresh(sched)
    return sched


def get_or_create_match(db, schedule: Schedule, week_start: date) -> Match:
    ws = fmt_week(week_start)
    m = db.query(Match).filter(Match.schedule_id == schedule.id, Match.week_start == ws).first()
    if m:
        return m

    weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][schedule.weekday]
    start_text = f"{weekday_name} {schedule.time_text}"

    m = Match(
        schedule_id=schedule.id,
        week_start=ws,
        start_time_text=start_text,
        location=schedule.location,
        total_cost=schedule.default_cost,
        is_open=True,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def promote_waitlist_if_needed(db, match: Match):
    sched = match.schedule
    if not sched.waitlist_enabled:
        return

    confirmed_count = db.query(Signup).filter(Signup.match_id == match.id, Signup.status == "confirmed").count()
    if confirmed_count >= sched.max_players:
        return

    nxt = (
        db.query(Signup)
        .filter(Signup.match_id == match.id, Signup.status == "waitlist")
        .order_by(Signup.created_at.asc())
        .first()
    )
    if nxt:
        nxt.status = "confirmed"
        db.commit()

def match_day_for_week(week_start: date, weekday: int) -> date:
    # week_start is Monday; weekday: 0=Mon..6=Sun (Fri=4)
    return week_start + timedelta(days=weekday)

def friday_badge(match_day: date, today: date) -> str | None:
    # badges for UX
    if match_day == today:
        return "Today"
    if match_day == today + timedelta(days=1):
        return "Tomorrow"
    # "This Friday" means same calendar week as today's Friday (based on schedule weekday)
    # We'll do simple: within next 7 days.
    if today < match_day <= today + timedelta(days=7):
        return "This Friday"
    if today + timedelta(days=7) < match_day <= today + timedelta(days=14):
        return "Next Friday"
    return None

# -------------------- lineup image --------------------
FORMATION_433 = ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CAM", "LW", "ST", "RW"]


def snake_split(players):
    # players: list of dict {name, level, position}
    players = sorted(players, key=lambda x: (-x["level"], x["name"].lower()))
    a, b = [], []
    flip = False
    for i, p in enumerate(players):
        if not flip:
            (a if i % 2 == 0 else b).append(p)
        else:
            (b if i % 2 == 0 else a).append(p)
        if (i + 1) % 4 == 0:
            flip = not flip
    return a, b


def assign_positions(team, formation):
    remaining = team[:]
    assigned = []
    for slot in formation:
        pick = None
        for p in remaining:
            if (p.get("position", "").upper() == slot):
                pick = p
                break
        if not pick:
            pick = remaining[0] if remaining else {"name": "—", "level": 0, "position": slot}
        if pick in remaining:
            remaining.remove(pick)
        assigned.append({**pick, "slot": slot})
    return assigned[: len(team)]


def make_lineup_image(teamA, teamB, week, when_text):
    W, H = 1400, 800
    img = Image.new("RGB", (W, H), (22, 110, 68))
    d = ImageDraw.Draw(img)

    line = (240, 255, 245)

    # fonts (bigger!)
    title_font = load_font(26)
    name_font  = load_font(22)   # bigger names
    pos_font   = load_font(18)   # position inside circle

    # pitch bounds
    L, T, R, B = 30, 60, W - 30, H - 30
    d.rectangle([L, T, R, B], outline=line, width=4)

    # halfway line + center circle
    midx = (L + R) // 2
    midy = (T + B) // 2
    d.line([midx, T, midx, B], fill=line, width=4)
    d.ellipse([midx - 90, midy - 90, midx + 90, midy + 90], outline=line, width=4)

    # --- penalty / goalie boxes ---
    pitch_w = R - L
    pitch_h = B - T

    # realistic-ish proportions
    pen_w = int(pitch_w * 0.18)         # penalty area depth
    pen_h = int(pitch_h * 0.58)         # penalty area height

    six_w = int(pitch_w * 0.07)         # 6-yard box depth
    six_h = int(pitch_h * 0.30)         # 6-yard box height

    # left penalty area + 6 yard
    lp_top = midy - pen_h // 2
    lp_bot = midy + pen_h // 2
    d.rectangle([L, lp_top, L + pen_w, lp_bot], outline=line, width=4)

    ls_top = midy - six_h // 2
    ls_bot = midy + six_h // 2
    d.rectangle([L, ls_top, L + six_w, ls_bot], outline=line, width=4)

    # right penalty area + 6 yard
    rp_top = lp_top
    rp_bot = lp_bot
    d.rectangle([R - pen_w, rp_top, R, rp_bot], outline=line, width=4)
    d.rectangle([R - six_w, ls_top, R, ls_bot], outline=line, width=4)

    # penalty spots
    left_spot_x = L + int(pen_w * 0.6)
    right_spot_x = R - int(pen_w * 0.6)
    spot_r = 4
    d.ellipse([left_spot_x - spot_r, midy - spot_r, left_spot_x + spot_r, midy + spot_r], fill=line)
    d.ellipse([right_spot_x - spot_r, midy - spot_r, right_spot_x + spot_r, midy + spot_r], fill=line)

    # title
    d.text(
        (40, 18),
        f"{week} · {when_text}",
        fill=(255, 255, 255),
        font=title_font
    )


    # coordinates (same idea)
    coords_left = {
        "GK": (180, 400),
        "LB": (320, 190), "CB1": (320, 340), "CB2": (320, 460), "RB": (320, 610),
        "CM1": (520, 310), "CM2": (520, 490), "CAM": (620, 400),
        "LW": (760, 220), "ST": (780, 400), "RW": (760, 580),
    }
    coords_right = {k: (W - x, y) for k, (x, y) in coords_left.items()}

    slot_to_key = {"GK": "GK", "LB": "LB", "RB": "RB", "CAM": "CAM", "LW": "LW", "ST": "ST", "RW": "RW"}
    usedA, usedB = set(), set()

    def coord_for(slot, side):
        if slot == "CB":
            base = "CB1" if ("CB1" not in (usedA if side == "A" else usedB)) else "CB2"
        elif slot == "CM":
            base = "CM1" if ("CM1" not in (usedA if side == "A" else usedB)) else "CM2"
        else:
            base = slot_to_key.get(slot, "CM2")

        if side == "A":
            usedA.add(base)
            return coords_left[base]
        usedB.add(base)
        return coords_right[base]

    def draw_text_centered(x, y, text, font, fill):
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        d.text((x - tw // 2, y - th // 2), text, fill=fill, font=font)

    def draw_player(x, y, name, pos, fill):
        r = 38
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=(255, 255, 255), width=4)

        # position centered inside circle
        draw_text_centered(x, y, pos, pos_font, (10, 10, 10))

        # name below (ALL CAPS, pure black, no outline)
        nm = name.upper()[:16]
        draw_text_centered(x, y + 52, nm, name_font, (0, 0, 0))



    # assign positions (keep your existing assign_positions)
    a_assigned = assign_positions(teamA, FORMATION_433)
    b_assigned = assign_positions(teamB, FORMATION_433)

    d.text((60, 90), "TEAM A", fill=(255, 255, 255), font=title_font)
    d.text((W - 220, 90), "TEAM B", fill=(255, 255, 255), font=title_font)

    for p in a_assigned:
        x, y = coord_for(p["slot"], "A")
        draw_player(x, y, p["name"], p["slot"], (92, 170, 255))

    for p in b_assigned:
        x, y = coord_for(p["slot"], "B")
        draw_player(x, y, p["name"], p["slot"], (120, 255, 170))

    return img



# -------------------- public --------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request, week: str | None = None):
    db = SessionLocal()
    try:
        user = get_current_user(request)
        sched = get_default_schedule(db)

        today = datetime.utcnow().date()
        week_start = monday_of_week(today)
        if week:
            try:
                week_start = parse_week_start(week)
            except Exception:
                pass

        match = get_or_create_match(db, sched, week_start)

        signups = db.query(Signup).filter(Signup.match_id == match.id).order_by(Signup.created_at.asc()).all()
        confirmed = [s for s in signups if s.status == "confirmed"]
        waitlist = [s for s in signups if s.status == "waitlist"]

        today = datetime.utcnow().date()

        weeks = []
        for i in range(-4, 5):
            ws = week_start + timedelta(days=7 * i)
            md = match_day_for_week(ws, sched.weekday)
            badge = friday_badge(md, today)
            label = md.strftime("%a %d %b %Y")  # Fri 19 Dec 2025
            if badge:
                label = f"{badge} · {label}"
            weeks.append({"value": fmt_week(ws), "label": label})

        cpp = cost_per_player_pence(match.total_cost, len(confirmed))

        selected_match_day = match_day_for_week(week_start, sched.weekday)
        match_date_text = selected_match_day.strftime("%a %d %b %Y")

        tpl = templates.get_template("index.html")
        return tpl.render(
            schedule=sched,
            match=match,
            weeks=weeks,
            match_date_text=match_date_text,
            selected_week=fmt_week(week_start),
            confirmed=confirmed,
            waitlist=waitlist,
            confirmed_count=len(confirmed),
            waitlist_count=len(waitlist),
            max_players=sched.max_players,
            cost_per=pence_to_gbp_str(cpp),
            user=user,
            request_path=request_path(request),
        )
    finally:
        db.close()


@app.post("/join")
def join(request: Request, week: str = Form(...)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url=f"/login?next=/?week={week}", status_code=303)

    db = SessionLocal()
    try:
        sched = get_default_schedule(db)
        match = get_or_create_match(db, sched, parse_week_start(week))

        if not match.is_open:
            return RedirectResponse(url=f"/?week={week}&closed=1", status_code=303)

        confirmed_count = db.query(Signup).filter(Signup.match_id == match.id, Signup.status == "confirmed").count()
        status = "confirmed"
        if confirmed_count >= sched.max_players and sched.waitlist_enabled:
            status = "waitlist"

        s = Signup(
            match_id=match.id,
            user_id=user.id,
            name=user.display_name,
            token=secrets.token_urlsafe(16),
            status=status
        )
        db.add(s)
        db.commit()
        return RedirectResponse(url=f"/?week={week}&joined=1", status_code=303)

    except IntegrityError:
        db.rollback()
        return RedirectResponse(url=f"/?week={week}&duplicate=1", status_code=303)
    finally:
        db.close()


@app.post("/leave")
def leave(request: Request, week: str = Form(...)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url=f"/login?next=/?week={week}", status_code=303)

    db = SessionLocal()
    try:
        sched = get_default_schedule(db)
        match = get_or_create_match(db, sched, parse_week_start(week))

        if not match.is_open:
            return RedirectResponse(url=f"/?week={week}&closed=1", status_code=303)

        s = db.query(Signup).filter(Signup.match_id == match.id, Signup.user_id == user.id).first()
        if not s:
            return RedirectResponse(url=f"/?week={week}&notfound=1", status_code=303)

        db.delete(s)
        db.commit()

        match = db.query(Match).filter(Match.id == match.id).first()
        promote_waitlist_if_needed(db, match)

        return RedirectResponse(url=f"/?week={week}&left=1", status_code=303)
    finally:
        db.close()


@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    db = SessionLocal()
    try:
        user = get_current_user(request)
        sched = get_default_schedule(db)
        matches = (
            db.query(Match)
            .filter(Match.schedule_id == sched.id)
            .order_by(Match.week_start.desc())
            .limit(52)
            .all()
        )
        tpl = templates.get_template("history.html")
        return tpl.render(matches=matches, user=user, request_path=request_path(request))
    finally:
        db.close()


@app.get("/month/{yyyymm}", response_class=HTMLResponse)
def month_totals(request: Request, yyyymm: str):
    db = SessionLocal()
    try:
        user = get_current_user(request)
        sched = get_default_schedule(db)
        start, end = month_window(yyyymm)

        matches = db.query(Match).filter(Match.schedule_id == sched.id).all()
        totals = {}  # name -> pence owed

        for m in matches:
            ws = parse_week_start(m.week_start)
            if not (start <= ws < end):
                continue

            signups = db.query(Signup).filter(Signup.match_id == m.id).all()
            confirmed = [s for s in signups if s.status == "confirmed"]
            cpp = cost_per_player_pence(m.total_cost, len(confirmed))
            for s in confirmed:
                totals[s.name] = totals.get(s.name, 0) + cpp

        items = sorted(totals.items(), key=lambda x: (-x[1], x[0].lower()))
        items_fmt = [(n, pence_to_gbp_str(p)) for n, p in items]

        tpl = templates.get_template("month.html")
        return tpl.render(yyyymm=yyyymm, items=items_fmt, user=user, request_path=request_path(request))
    finally:
        db.close()


# -------------------- auth --------------------
@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, next: str = "/"):
    tpl = templates.get_template("signup.html")
    return tpl.render(
        next=next,
        invite_required=bool(INVITE_CODE),
        user=get_current_user(request),
        request_path=request_path(request),
    )


@app.post("/signup")
def signup(next: str = Form("/"), name: str = Form(...), pin: str = Form(...), invite: str = Form("")):
    if INVITE_CODE and invite.strip() != INVITE_CODE:
        return RedirectResponse(url=f"/signup?next={next}&badinvite=1", status_code=303)

    db = SessionLocal()
    try:
        clean = normalize_name(name)
        ph = pin_hash(pin)

        u = User(display_name=clean, pin_hash=ph)
        # make first user admin automatically
        if db.query(User).count() == 0:
            u.is_admin = True

        db.add(u)
        db.commit()
        db.refresh(u)

        session = session_signer.dumps({"uid": u.id})
        resp = RedirectResponse(url=next or "/me", status_code=303)
        resp.set_cookie("fb_session", session, max_age=60 * 60 * 24 * 365, httponly=True, samesite="lax")
        return resp

    except IntegrityError:
        db.rollback()
        return RedirectResponse(url=f"/signup?next={next}&exists=1", status_code=303)
    finally:
        db.close()


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    tpl = templates.get_template("login.html")
    return tpl.render(
        next=next,
        user=get_current_user(request),
        request_path=request_path(request),
    )


@app.post("/login")
def login(next: str = Form("/"), name: str = Form(...), pin: str = Form(...)):
    db = SessionLocal()
    try:
        clean = normalize_name(name)
        u = db.query(User).filter(User.display_name == clean).first()
        if not u or not verify_pin(pin, u.pin_hash):
            return RedirectResponse(url=f"/login?next={next}&bad=1", status_code=303)

        session = session_signer.dumps({"uid": u.id})
        resp = RedirectResponse(url=next or "/me", status_code=303)
        resp.set_cookie("fb_session", session, max_age=60 * 60 * 24 * 365, httponly=True, samesite="lax")
        return resp
    finally:
        db.close()


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("fb_session")
    return resp


# -------------------- dashboard --------------------
@app.get("/me", response_class=HTMLResponse)
def me(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login?next=/me", status_code=303)

    db = SessionLocal()
    try:
        now = datetime.utcnow().date()
        yyyymm = f"{now.year:04d}-{now.month:02d}"
        start, end = month_window(yyyymm)

        # played + owed (this month)
        matches = db.query(Match).order_by(Match.week_start.desc()).all()
        played = []
        owed_month = 0

        for m in matches:
            ws = parse_week_start(m.week_start)
            if not (start <= ws < end):
                continue

            signups = db.query(Signup).filter(Signup.match_id == m.id).all()
            confirmed = [s for s in signups if s.status == "confirmed"]
            if not any(s.user_id == user.id for s in confirmed):
                continue

            played.append(m)
            owed_month += cost_per_player_pence(m.total_cost, len(confirmed))

        # payments month
        month_payments = db.query(Payment).filter(
            Payment.user_id == user.id,
            Payment.paid_at >= datetime(start.year, start.month, start.day),
            Payment.paid_at < datetime(end.year, end.month, end.day),
        ).order_by(Payment.paid_at.desc()).all()

        paid_month = sum(p.amount_pence for p in month_payments)

        # all time owed (confirmed games)
        all_owed = 0
        all_matches = db.query(Match).all()
        for m in all_matches:
            signups = db.query(Signup).filter(Signup.match_id == m.id).all()
            confirmed = [s for s in signups if s.status == "confirmed"]
            if any(s.user_id == user.id for s in confirmed):
                all_owed += cost_per_player_pence(m.total_cost, len(confirmed))

        all_paid = sum(p.amount_pence for p in db.query(Payment).filter(Payment.user_id == user.id).all())

        payments_fmt = [(pence_to_gbp_str(p.amount_pence), p.paid_at.strftime("%Y-%m-%d"), p.note) for p in month_payments]

        tpl = templates.get_template("me.html")
        return tpl.render(
            user=user,
            request_path=request_path(request),

            yyyymm=yyyymm,
            played=played,

            owed_month=pence_to_gbp_str(owed_month),
            paid_month=pence_to_gbp_str(paid_month),
            remaining_month=pence_to_gbp_str(owed_month - paid_month),

            owed_all=pence_to_gbp_str(all_owed),
            paid_all=pence_to_gbp_str(all_paid),
            remaining_all=pence_to_gbp_str(all_owed - all_paid),

            payments_fmt=payments_fmt,
        )
    finally:
        db.close()


@app.get("/lineup.png")
def lineup_png(week: str):
    db = SessionLocal()
    try:
        sched = get_default_schedule(db)
        match = get_or_create_match(db, sched, parse_week_start(week))

        signups = db.query(Signup).filter(Signup.match_id == match.id, Signup.status == "confirmed").all()
        players = []
        for s in signups:
            if not s.user_id:
                continue
            u = db.query(User).filter(User.id == s.user_id).first()
            if not u:
                continue
            players.append({
                "name": u.display_name,
                "level": int(u.level or 5),
                "position": (u.position or "").upper(),
            })

        teamA, teamB = snake_split(players)
        ws = parse_week_start(match.week_start)           # convert "YYYY-MM-DD" -> date
        match_day = ws + timedelta(days=sched.weekday)    # now timedelta works
        match_day_text = match_day.strftime("%a %d %b %Y")

        img = make_lineup_image(
            teamA,
            teamB,
            week=match_day_text,
            when_text=sched.time_text
        )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    finally:
        db.close()

@app.post("/me/pay")
def me_pay(request: Request, amount: str = Form(...), note: str = Form("")):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login?next=/me", status_code=303)

    db = SessionLocal()
    try:
        amt = pounds_to_pence(amount)
        if amt <= 0:
            return RedirectResponse(url="/me?badamount=1", status_code=303)

        db.add(Payment(user_id=user.id, amount_pence=amt, note=(note or "").strip()))
        db.commit()
        return RedirectResponse(url="/me?paid=1", status_code=303)
    finally:
        db.close()


# -------------------- admin --------------------
@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    user = require_admin_user(request)

    db = SessionLocal()
    try:
        sched = get_default_schedule(db)
        today_week = fmt_week(monday_of_week(datetime.utcnow().date()))

        tpl = templates.get_template("admin.html")
        return tpl.render(
            schedule=sched,
            today_week=today_week,
            user=user,
            request_path=request_path(request),
        )
    finally:
        db.close()


@app.post("/admin/schedule")
def admin_schedule(
    request: Request,
    title: str = Form(...),
    weekday: int = Form(...),
    time_text: str = Form(...),
    location: str = Form(...),
    default_cost: int = Form(...),
    max_players: int = Form(...),
    waitlist_enabled: str = Form(...),
):
    require_admin_user(request)

    db = SessionLocal()
    try:
        sched = get_default_schedule(db)
        sched.title = (title.strip() or sched.title)
        sched.weekday = max(0, min(6, int(weekday)))
        sched.time_text = (time_text.strip() or sched.time_text)
        sched.location = (location.strip() or sched.location)
        sched.default_cost = max(0, int(default_cost))
        sched.max_players = max(1, int(max_players))
        sched.waitlist_enabled = (waitlist_enabled == "1")
        db.commit()
        return RedirectResponse(url="/admin?saved=1", status_code=303)
    finally:
        db.close()


@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    user = require_admin_user(request)

    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.display_name.asc()).all()
        tpl = templates.get_template("admin_users.html")
        return tpl.render(
            users=users,
            user=user,
            request_path=request_path(request),
        )
    finally:
        db.close()


@app.post("/admin/users/update")
def admin_users_update(
    request: Request,
    user_id: int = Form(...),
    level: int = Form(...),
    position: str = Form(...),
):
    require_admin_user(request)

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == user_id).first()
        if u:
            u.level = max(1, min(10, int(level)))
            u.position = (position or "").strip().upper()
            db.commit()
        return RedirectResponse(url="/admin/users", status_code=303)
    finally:
        db.close()


@app.get("/admin/week", response_class=HTMLResponse)
def admin_week(request: Request, week: str):
    user = require_admin_user(request)

    db = SessionLocal()
    try:
        sched = get_default_schedule(db)
        match = get_or_create_match(db, sched, parse_week_start(week))

        signups = (
            db.query(Signup)
            .filter(Signup.match_id == match.id)
            .order_by(Signup.created_at.asc())
            .all()
        )
        confirmed = [s for s in signups if s.status == "confirmed"]
        waitlist = [s for s in signups if s.status == "waitlist"]

        tpl = templates.get_template("admin_week.html")
        return tpl.render(
            match=match,
            confirmed=confirmed,
            waitlist=waitlist,
            user=user,
            request_path=request_path(request),
        )
    finally:
        db.close()


@app.get("/admin/lineup.png")
def admin_lineup_png(request: Request, week: str):
    require_admin_user(request)

    db = SessionLocal()
    try:
        sched = get_default_schedule(db)
        match = get_or_create_match(db, sched, parse_week_start(week))

        signups = db.query(Signup).filter(
            Signup.match_id == match.id,
            Signup.status == "confirmed"
        ).all()

        players = []
        for s in signups:
            if not s.user_id:
                continue
            u = db.query(User).filter(User.id == s.user_id).first()
            if not u:
                continue
            players.append({
                "name": u.display_name,
                "level": int(u.level or 5),
                "position": (u.position or "").upper(),
            })

        teamA, teamB = snake_split(players)

        # show match DAY in the header (Fri 19 Dec 2025), not Monday
        ws = parse_week_start(match.week_start)
        match_day = ws + timedelta(days=sched.weekday)
        match_day_text = match_day.strftime("%a %d %b %Y")

        img = make_lineup_image(teamA, teamB, week=match_day_text, when_text=sched.time_text)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    finally:
        db.close()
