"""Complete FRMS portal routes, separated from the recognition engine."""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from uuid import uuid4
import time

from flask import abort, current_app, flash, jsonify, redirect, render_template, request, session, url_for
import os
from werkzeug.utils import secure_filename

STAFF = ("super_admin", "pdrm_officer")
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def register_portal(app):
    def db(): return current_app.extensions["system_db"]
    def roles(*allowed):
        def deco(view):
            @wraps(view)
            def inner(*args, **kwargs):
                if not session.get("user_id"): return redirect(url_for("login", next=request.path))
                if allowed and session.get("role") not in allowed: abort(403)
                return view(*args, **kwargs)
            return inner
        return deco
    def audit(action, module, text):
        with db().cursor() as c: c.execute("INSERT INTO audit_logs (user_id,action,module,description,created_at) VALUES (%s,%s,%s,%s,now())", (session.get("user_id"),action,module,text))
    def get_case(case_id):
        with db().cursor() as c:
            c.execute("SELECT m.*,u.name AS creator_name FROM missing_persons m JOIN users u ON u.id=m.created_by WHERE m.id=%s", (case_id,)); row=c.fetchone()
        if not row: abort(404)
        if session["role"] == "public_user" and row["created_by"] != session["user_id"]: abort(403)
        return row

    @app.get("/dashboard/admin")
    @roles("super_admin")
    def admin_dashboard(): return dashboard("admin")
    @app.get("/dashboard/pdrm")
    @roles(*STAFF)
    def pdrm_dashboard(): return dashboard("pdrm")
    @app.get("/dashboard/patrol")
    @roles("super_admin", "patrol_unit")
    def patrol_dashboard(): return dashboard("patrol")
    @app.get("/dashboard/public")
    @roles("super_admin", "public_user")
    def public_dashboard(): return dashboard("public")
    def dashboard(kind):
        with db().cursor() as c:
            c.execute("SELECT count(*) AS n FROM missing_persons WHERE status='active'"); active=c.fetchone()["n"]
            c.execute("SELECT count(*) AS n FROM alerts WHERE status='new'"); new_alerts=c.fetchone()["n"]
            c.execute("SELECT count(*) AS n FROM verification_requests WHERE status='pending'"); pending=c.fetchone()["n"]
            c.execute("SELECT count(*) AS n FROM cameras WHERE status='online'"); online=c.fetchone()["n"]
            c.execute("SELECT a.*,m.full_name FROM alerts a JOIN missing_persons m ON m.id=a.missing_person_id WHERE (%s <> 'patrol' OR a.assigned_to=%s) ORDER BY a.created_at DESC LIMIT 8", (kind,session["user_id"])); alerts=c.fetchall()
            c.execute("SELECT case_number,full_name,status,created_at FROM missing_persons WHERE (%s <> 'public' OR created_by=%s) ORDER BY created_at DESC LIMIT 8", (kind,session["user_id"])); cases=c.fetchall()
        return render_template("frms/dashboard.html", kind=kind, stats={"active_cases":active,"new_alerts":new_alerts,"pending_verifications":pending,"online_cameras":online}, alerts=alerts,cases=cases)

    @app.route("/missing-persons", methods=["GET","POST"])
    @roles(*STAFF, "public_user")
    def missing_persons():
        if request.method == "POST":
            name=request.form.get("full_name", "").strip(); location=request.form.get("last_seen_location", "").strip()
            if not name or not location: flash("Full name and last seen location are required.","error"); return redirect(request.url)
            status="pending" if session["role"] == "public_user" else "active"; number=f"MP-{time.strftime('%Y')}-{uuid4().hex[:4].upper()}"
            with db().cursor() as c:
                c.execute("INSERT INTO missing_persons (case_number,full_name,gender,dob,national_id,description,last_seen_location,last_seen_date,status,created_by,verified_by,verified_at,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,CASE WHEN %s='active' THEN now() END,now(),now()) RETURNING id", (number,name,request.form.get("gender") or None,request.form.get("dob") or None,request.form.get("national_id") or None,request.form.get("description") or None,location,request.form.get("last_seen_date") or None,status,session["user_id"],session["user_id"] if status=="active" else None,status)); cid=c.fetchone()["id"]
                c.execute("INSERT INTO missing_person_timeline (missing_person_id,event_type,description,recorded_by,created_at) VALUES (%s,'case_created',%s,%s,now())",(cid,f"Case {number} created with status {status}.",session["user_id"]))
            audit("create","missing_persons",f"Created {number}"); return redirect(url_for("missing_person_detail", case_id=cid))
        q=request.args.get("q","").strip(); status=request.args.get("status","")
        sql="SELECT m.*,u.name AS creator_name FROM missing_persons m JOIN users u ON u.id=m.created_by WHERE (%s='' OR m.full_name ILIKE %s OR m.case_number ILIKE %s) AND (%s='' OR m.status::text=%s)"; params=(q,f"%{q}%",f"%{q}%",status,status)
        if session["role"]=="public_user": sql+=" AND m.created_by=%s"; params+= (session["user_id"],)
        with db().cursor() as c: c.execute(sql+" ORDER BY m.created_at DESC",params); rows=c.fetchall()
        return render_template("frms/cases.html", cases=rows)

    @app.route("/missing-persons/<int:case_id>", methods=["GET","POST"])
    @roles(*STAFF,"public_user")
    def missing_person_detail(case_id):
        case=get_case(case_id)
        if request.method=="POST":
            if session["role"] not in STAFF: abort(403)
            with db().cursor() as c: c.execute("UPDATE missing_persons SET full_name=%s,gender=%s,dob=%s,national_id=%s,description=%s,last_seen_location=%s,last_seen_date=%s,updated_at=now() WHERE id=%s",tuple(request.form.get(k) or None for k in ("full_name","gender","dob","national_id","description","last_seen_location","last_seen_date"))+(case_id,))
            audit("update","missing_persons",f"Updated {case['case_number']}"); flash("Case updated.","success"); return redirect(request.url)
        with db().cursor() as c:
            c.execute("SELECT * FROM missing_person_photos WHERE missing_person_id=%s ORDER BY is_primary DESC,created_at DESC",(case_id,)); photos=c.fetchall()
            c.execute("SELECT t.*,u.name FROM missing_person_timeline t LEFT JOIN users u ON u.id=t.recorded_by WHERE missing_person_id=%s ORDER BY created_at DESC",(case_id,)); timeline=c.fetchall()
        return render_template("frms/case_detail.html", case=case,photos=photos,timeline=timeline)

    @app.post("/missing-persons/<int:case_id>/activate")
    @roles(*STAFF)
    def activate_case(case_id):
        case=get_case(case_id)
        with db().cursor() as c:
            c.execute("UPDATE missing_persons SET status='active',verified_by=%s,verified_at=now(),updated_at=now() WHERE id=%s",(session["user_id"],case_id)); c.execute("INSERT INTO missing_person_timeline (missing_person_id,event_type,description,recorded_by,created_at) VALUES (%s,'verified_active','PDRM officer verified and activated the case.',%s,now())",(case_id,session["user_id"]))
        audit("activate","missing_persons",case["case_number"]); return redirect(url_for("missing_person_detail",case_id=case_id))

    @app.post("/missing-persons/<int:case_id>/photos")
    @roles(*STAFF)
    def case_photo(case_id):
        get_case(case_id); f=request.files.get("photo")
        if not f or "." not in f.filename or f.filename.rsplit(".",1)[1].lower() not in IMAGE_EXTENSIONS: abort(400)
        folder=Path(current_app.config["UPLOAD_FOLDER"])/"cases"; folder.mkdir(parents=True,exist_ok=True); filename=f"{case_id}_{uuid4().hex}.{f.filename.rsplit('.',1)[1].lower()}"; f.save(folder/filename)
        with db().cursor() as c:
            primary=request.form.get("is_primary")=="on"
            if primary: c.execute("UPDATE missing_person_photos SET is_primary=false WHERE missing_person_id=%s",(case_id,))
            c.execute("INSERT INTO missing_person_photos (missing_person_id,path,type,is_primary,uploaded_by,created_at,updated_at) VALUES (%s,%s,'reference',%s,%s,now(),now())",(case_id,f"uploads/cases/{filename}",primary,session["user_id"]))
        return redirect(url_for("missing_person_detail",case_id=case_id))

    @app.get("/detections")
    @roles(*STAFF)
    def detections():
        with db().cursor() as c: c.execute("SELECT d.*,m.full_name,c.name AS camera_name FROM detections d JOIN missing_persons m ON m.id=d.reference_id JOIN cameras c ON c.id=d.camera_id ORDER BY d.detected_at DESC"); rows=c.fetchall()
        return render_template("frms/detections.html", detections=rows)
    @app.get("/detections/<int:detection_id>")
    @roles(*STAFF)
    def detection_detail(detection_id):
        with db().cursor() as c: c.execute("SELECT d.*,m.full_name,c.name AS camera_name,a.id AS alert_db_id,a.alert_id FROM detections d JOIN missing_persons m ON m.id=d.reference_id JOIN cameras c ON c.id=d.camera_id LEFT JOIN alerts a ON a.detection_id=d.id WHERE d.id=%s",(detection_id,)); row=c.fetchone()
        if not row: abort(404)
        return render_template("frms/detection_detail.html", detection=row)

    @app.route("/alerts",methods=["GET"])
    @roles(*STAFF,"patrol_unit")
    def alerts():
        with db().cursor() as c: c.execute("SELECT a.*,m.full_name,u.name AS assignee FROM alerts a JOIN missing_persons m ON m.id=a.missing_person_id LEFT JOIN users u ON u.id=a.assigned_to WHERE (%s <> 'patrol_unit' OR a.assigned_to=%s) ORDER BY a.created_at DESC",(session["role"],session["user_id"])); rows=c.fetchall()
        return render_template("frms/alerts.html",alerts=rows)
    @app.route("/alerts/<int:alert_id>",methods=["GET","POST"])
    @roles(*STAFF,"patrol_unit")
    def alert_detail(alert_id):
        with db().cursor() as c:
            c.execute("SELECT a.*,m.full_name,d.similarity_score FROM alerts a JOIN missing_persons m ON m.id=a.missing_person_id JOIN detections d ON d.id=a.detection_id WHERE a.id=%s",(alert_id,)); row=c.fetchone()
            if not row: abort(404)
            if session["role"]=="patrol_unit" and row["assigned_to"]!=session["user_id"]: abort(403)
            if request.method=="POST":
                status=request.form.get("status")
                if session["role"]=="patrol_unit" and status not in {"under_review","closed"}: abort(400)
                if status not in {"under_review","escalated","closed","false_positive"}: abort(400)
                c.execute("UPDATE alerts SET status=%s,resolved_by=CASE WHEN %s='closed' THEN %s ELSE resolved_by END,resolved_at=CASE WHEN %s='closed' THEN now() ELSE resolved_at END,updated_at=now() WHERE id=%s",(status,status,session["user_id"],status,alert_id)); c.execute("INSERT INTO alert_timeline (alert_id,event_type,description,performed_by,created_at) VALUES (%s,%s,%s,%s,now())",(alert_id,status,request.form.get("notes") or "Alert status updated.",session["user_id"]))
                return redirect(request.url)
            c.execute("SELECT * FROM alert_timeline WHERE alert_id=%s ORDER BY created_at DESC",(alert_id,)); timeline=c.fetchall()
            c.execute("SELECT id,name FROM users WHERE role='patrol_unit' AND is_active ORDER BY name"); patrols=c.fetchall()
        return render_template("frms/alert_detail.html",alert=row,timeline=timeline,patrols=patrols)
    @app.post("/alerts/<int:alert_id>/assign")
    @roles(*STAFF)
    def assign_alert(alert_id):
        patrol=request.form.get("patrol_user_id",type=int)
        with db().cursor() as c: c.execute("UPDATE alerts SET assigned_to=%s,assigned_at=now(),status='escalated',updated_at=now() WHERE id=%s",(patrol,alert_id)); c.execute("INSERT INTO alert_timeline (alert_id,event_type,description,performed_by,created_at) VALUES (%s,'assigned','Alert assigned to patrol unit.',%s,now())",(alert_id,session["user_id"]))
        return redirect(url_for("alert_detail",alert_id=alert_id))

    @app.route("/verifications",methods=["GET"])
    @roles(*STAFF)
    def verifications():
        with db().cursor() as c: c.execute("SELECT v.*,m.full_name,a.alert_id FROM verification_requests v JOIN alerts a ON a.id=v.alert_id JOIN missing_persons m ON m.id=a.missing_person_id ORDER BY v.created_at DESC"); rows=c.fetchall()
        return render_template("frms/verifications.html",verifications=rows)
    @app.post("/verifications/<int:request_id>")
    @roles(*STAFF)
    def review_verification(request_id):
        status=request.form.get("status")
        if status not in {"confirmed","rejected","insufficient_data"}: abort(400)
        with db().cursor() as c: c.execute("UPDATE verification_requests SET status=%s,jpn_notes=%s,police_notes=%s,reviewed_by=%s,verified_at=now(),updated_at=now() WHERE id=%s",(status,request.form.get("jpn_notes"),request.form.get("police_notes"),session["user_id"],request_id))
        return redirect(url_for("verifications"))

    @app.route("/cameras",methods=["GET","POST"])
    @roles(*STAFF)
    def camera_list():
        if request.method=="POST":
            with db().cursor() as c: c.execute("INSERT INTO cameras (camera_id,name,location,stream_url,stream_type,status,registered_by,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,now(),now())",(request.form["camera_id"],request.form["name"],request.form.get("location"),request.form.get("stream_url"),request.form.get("stream_type","rtsp"),request.form.get("status","offline"),session["user_id"]))
            return redirect(request.url)
        with db().cursor() as c: c.execute("SELECT * FROM cameras ORDER BY name"); rows=c.fetchall()
        return render_template("frms/cameras.html",cameras=rows)

    @app.route("/users",methods=["GET","POST"])
    @roles("super_admin")
    def users():
        if request.method=="POST":
            from werkzeug.security import generate_password_hash
            with db().cursor() as c: c.execute("INSERT INTO users (name,email,password,role,agency,phone,is_active,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,true,now(),now())",(request.form["name"],request.form["email"],generate_password_hash(request.form["password"]),request.form["role"],request.form.get("agency"),request.form.get("phone")))
            return redirect(request.url)
        with db().cursor() as c: c.execute("SELECT id,name,email,role,agency,phone,is_active,created_at FROM users ORDER BY name"); rows=c.fetchall()
        return render_template("frms/users.html",users=rows)
    @app.post("/users/<int:user_id>/toggle")
    @roles("super_admin")
    def toggle_user(user_id):
        if user_id==session["user_id"]: abort(400)
        with db().cursor() as c: c.execute("UPDATE users SET is_active=NOT is_active,updated_at=now() WHERE id=%s",(user_id,))
        return redirect(url_for("users"))

    @app.get("/notifications")
    @roles("super_admin","pdrm_officer","patrol_unit","public_user")
    def notifications():
        with db().cursor() as c: c.execute("SELECT * FROM notifications WHERE notifiable_type='users' AND notifiable_id=%s ORDER BY created_at DESC",(session["user_id"],)); rows=c.fetchall()
        return render_template("frms/notifications.html",notifications=rows)
    @app.get("/audit-logs")
    @roles("super_admin")
    def audit_logs():
        with db().cursor() as c: c.execute("SELECT a.*,u.name FROM audit_logs a LEFT JOIN users u ON u.id=a.user_id ORDER BY a.created_at DESC LIMIT 500"); rows=c.fetchall()
        return render_template("frms/audit_logs.html",logs=rows)

    @app.route("/system-settings", methods=["GET", "POST"])
    @roles("super_admin")
    def system_settings():
        if request.method == "POST":
            key, value = request.form.get("key", "").strip(), request.form.get("value", "")
            if not key: abort(400)
            with db().cursor() as c: c.execute("INSERT INTO system_settings (key,value,type,description,created_at,updated_at) VALUES (%s,%s,%s,%s,now(),now()) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value,type=EXCLUDED.type,description=EXCLUDED.description,updated_at=now()", (key,value,request.form.get("type","string"),request.form.get("description")))
            audit("update","system_settings",f"Updated {key}"); return redirect(request.url)
        with db().cursor() as c: c.execute("SELECT * FROM system_settings ORDER BY key"); rows=c.fetchall()
        return render_template("frms/settings.html", settings=rows)

    @app.post("/api/verify-detection")
    def verify_detection_api():
        """Protected ingestion endpoint for a recognition worker or remote camera node."""
        key=os.environ.get("LARAVEL_API_KEY")
        if key and request.headers.get("X-API-Key") != key: abort(401)
        payload=request.get_json(silent=True) or request.form
        camera_id, case_number = payload.get("camera_id"), payload.get("case_number")
        try: score=float(payload.get("similarity_score"))
        except (TypeError,ValueError): return jsonify(error="similarity_score must be a number"),400
        if not camera_id or not case_number or not 0 <= score <= 100: return jsonify(error="camera_id, case_number and similarity_score (0-100) are required"),400
        with db().cursor() as c:
            c.execute("SELECT id FROM cameras WHERE camera_id=%s",(camera_id,)); camera=c.fetchone()
            c.execute("SELECT id,full_name FROM missing_persons WHERE case_number=%s AND status='active'",(case_number,)); person=c.fetchone()
            if not camera or not person: return jsonify(error="Active case or camera not found"),404
            det=f"DET-{time.strftime('%Y')}-{uuid4().hex[:6].upper()}"; alert=f"ALT-{time.strftime('%Y')}-{uuid4().hex[:6].upper()}"; vr=f"VRF-{time.strftime('%Y')}-{uuid4().hex[:6].upper()}"; requester=int(os.environ.get("PYTHON_SYSTEM_USER_ID","1"))
            c.execute("INSERT INTO detections (detection_id,camera_id,captured_image,reference_type,reference_id,similarity_score,status,detected_at,created_at,updated_at) VALUES (%s,%s,%s,'missing_person',%s,%s,'new',now(),now(),now()) RETURNING id",(det,camera["id"],payload.get("captured_image"),person["id"],score)); did=c.fetchone()["id"]
            c.execute("INSERT INTO alerts (alert_id,detection_id,missing_person_id,title,message,location,status,priority,created_at,updated_at) VALUES (%s,%s,%s,'CCTV face match',%s,%s,'new','high',now(),now()) RETURNING id",(alert,did,person["id"],f"{person['full_name']} matched at {score:.2f}%",payload.get("location"))); aid=c.fetchone()["id"]
            c.execute("INSERT INTO verification_requests (request_id,detection_id,alert_id,requested_by,status,created_at,updated_at) VALUES (%s,%s,%s,%s,'pending',now(),now())",(vr,did,aid,requester))
            c.execute("INSERT INTO alert_timeline (alert_id,event_type,description,performed_by,created_at) VALUES (%s,'created',%s,%s,now())",(aid,"Created by recognition worker.",requester))
        return jsonify(status="created",detection_id=det,alert_id=alert,verification_request_id=vr),201
