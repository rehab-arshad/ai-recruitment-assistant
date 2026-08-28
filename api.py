import os
import tempfile

from flask import Flask, jsonify, request

from pipeline import index_pdf, get_collection
from screen import match
from evaluate import evaluate

app = Flask(__name__)


@app.post("/evaluate")
def evaluate_candidate():
    if "resume" not in request.files:
        return jsonify({"error": "no 'resume' file uploaded"}), 400

    job_spec = request.form.get("job_spec", "").strip()
    if not job_spec:
        return jsonify({"error": "no 'job_spec' provided"}), 400

    upload = request.files["resume"]

    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, upload.filename)
    upload.save(path)

    try:
        collection, source = index_pdf(path, force=True)
    except ValueError as exc:
        return jsonify({"error": str(exc), "candidate": upload.filename}), 422

    hits = match(collection, job_spec, source, top_k=4)
    result = evaluate(job_spec, source, hits)
    result["retrieval_score"] = round(hits[0]["score"], 3) if hits else 0

    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({"ok": True, "chunks": get_collection().count()})


if __name__ == "__main__":
    app.run(port=5000)