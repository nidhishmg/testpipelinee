from env.models import GraderResult

def grader2(rows_passing: int, total_rows: int) -> GraderResult:
    score = rows_passing / total_rows if total_rows else 0.0
    return GraderResult(
        score=round(score, 4),
        breakdown={"rows_passing": rows_passing, "total_rows": total_rows},
        explanation=f"{rows_passing} of {total_rows} rows passed validation"
    )
