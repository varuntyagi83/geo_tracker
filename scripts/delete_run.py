# scripts/delete_run.py
import sys, datetime as dt
sys.path.append("src")
from geo_tracker.storage import get_session, init_db, Run, ModelCall, ExtractedMention, Sentiment, Grounding, MetricsDaily

rk = sys.argv[1] if len(sys.argv) > 1 else dt.datetime.utcnow().strftime("%Y%m%d") + "-default-DE-en"
init_db()
with get_session() as s:
    run = s.query(Run).filter(Run.run_key==rk).first()
    if not run:
        print("No such run:", rk); raise SystemExit(0)
    call_ids = [mc.id for mc in s.query(ModelCall.id).filter_by(run_id=run.id).all()]
    s.query(ExtractedMention).filter(ExtractedMention.call_id.in_(call_ids)).delete(synchronize_session=False)
    s.query(Sentiment).filter(Sentiment.call_id.in_(call_ids)).delete(synchronize_session=False)
    s.query(Grounding).filter(Grounding.call_id.in_(call_ids)).delete(synchronize_session=False)
    s.query(ModelCall).filter_by(run_id=run.id).delete(synchronize_session=False)
    # optional: delete today’s metrics
    s.query(MetricsDaily).filter(MetricsDaily.date==dt.date.today()).delete(synchronize_session=False)
    s.delete(run)
    s.commit()
    print("Deleted run:", rk)
