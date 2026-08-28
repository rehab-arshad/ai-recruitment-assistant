import os

from pipeline import index_pdf, get_collection, embed_texts

SUPPORTED = (".pdf", ".docx", ".txt", ".md")

JOB_SPEC = """Python developer with 2+ years experience.
Must have: REST APIs, PostgreSQL, Git.
Nice to have: Docker, AWS, machine learning."""


def index_folder(folder="documents"):
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED)
    ]
    if not paths:
        raise SystemExit(f"No supported files in {folder}/ — {', '.join(SUPPORTED)}")

    candidates = []
    for path in paths:
        try:
            _, source = index_pdf(path)
            candidates.append(source)
        except ValueError as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")

    return candidates


def match(collection, job_spec, candidate, top_k=3):
    q_vec = embed_texts([job_spec])[0]

    res = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k,
        where={"source": candidate},
    )

    return [
        {"text": text, "score": 1 - dist}
        for text, dist in zip(res["documents"][0], res["distances"][0])
    ]