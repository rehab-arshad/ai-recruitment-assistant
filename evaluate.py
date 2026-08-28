import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from pipeline import get_collection
from screen import JOB_SPEC, index_folder, match

load_dotenv()

LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
)

EVAL_PROMPT = """You are a technical recruiter screening ONE candidate against a job spec.

Judge only from the resume excerpts provided. Never assume experience the resume
does not show.

Verdict rules:
- "shortlist" — every must-have requirement is evidenced in the resume.
- "review"    — the must-have SKILLS are evidenced, but something the resume
                cannot confirm is in doubt: years of experience, seniority, or
                whether the work was professional rather than academic. Route
                these to a human instead of rejecting them.
- "reject"    — one or more must-have skills is absent from the resume entirely.

Treat a missing skill as a gap. Treat an unverifiable duration as a reason for
"review", not "reject" — a resume that omits dates is not the same as a
candidate who lacks the experience.

Return ONLY a JSON object — no markdown fences, no commentary — with these keys:
  "verdict":   "shortlist", "review", or "reject"
  "score":     integer 0-100, how well the candidate meets the requirements
  "reasoning": two or three sentences justifying the verdict
  "evidence":  array of short quotes from the resume that support your assessment
  "gaps":      array of requirements the resume does not evidence"""


def _failed(candidate, why):
    return {
        "candidate": candidate,
        "verdict": "review",
        "score": 0,
        "reasoning": f"Evaluation failed: {why}",
        "evidence": [],
        "gaps": [],
    }


def evaluate(job_spec, candidate, hits):
    excerpts = "\n\n---\n\n".join(
        f"[{i}] {h['text']}" for i, h in enumerate(hits, start=1)
    )
    user_message = (
        f"JOB SPEC:\n{job_spec}\n\n"
        f"RESUME EXCERPTS ({candidate}):\n\n{excerpts}"
    )

    try:
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EVAL_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
    except Exception as exc:
        return _failed(candidate, str(exc))

    # rate limits and provider errors come back with no choices at all
    if not getattr(res, "choices", None):
        return _failed(candidate, getattr(res, "error", None) or "no choices returned")

    raw = res.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "verdict": "review",
            "score": 0,
            "reasoning": "Model returned unparseable output.",
            "evidence": [],
            "gaps": [],
            "raw": raw,
        }

    result["candidate"] = candidate
    return result


if __name__ == "__main__":
    candidates = index_folder()
    collection = get_collection()

    print(f"\nJOB SPEC:\n{JOB_SPEC}\n" + "=" * 70)

    results = []
    for candidate in candidates:
        hits = match(collection, JOB_SPEC, candidate, top_k=4)
        retrieval_score = hits[0]["score"] if hits else 0

        print(f"\nevaluating {candidate}...")
        result = evaluate(JOB_SPEC, candidate, hits)
        result["retrieval_score"] = round(retrieval_score, 3)
        results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 70)
    print(f"{'LLM':>4} {'RETR':>6}  {'VERDICT':<10} CANDIDATE")
    print("-" * 70)
    for r in results:
        print(f"{r['score']:>4} {r['retrieval_score']:>6}  "
              f"{r['verdict']:<10} {r['candidate']}")

    print("\n" + "=" * 70)
    for r in results:
        print(f"\n{r['candidate']}  —  {r['verdict'].upper()} ({r['score']}/100)")
        print(f"  {r['reasoning']}")
        if r.get("evidence"):
            print("  Evidence:")
            for e in r["evidence"]:
                print(f"    • {e}")
        if r.get("gaps"):
            print("  Gaps:")
            for g in r["gaps"]:
                print(f"    • {g}")