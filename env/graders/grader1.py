from env.models import GraderResult

def grader1(identified: int, fixed: int, total: int) -> GraderResult:
    id_score = identified / total if total else 0.0
    fix_score = fixed / total if total else 0.0
    score = (0.4 * id_score) + (0.6 * fix_score)
    return GraderResult(
        score=round(score, 4),
        breakdown={"identified": round(id_score, 4), "fixed": round(fix_score, 4)},
        explanation=f"Identified {identified}/{total}, Fixed {fixed}/{total}"
    )
