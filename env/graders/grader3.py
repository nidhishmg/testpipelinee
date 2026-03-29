from env.models import GraderResult

def grader3(diagnosis_score: float, fix_score: float, pii_score: float, validation_score: float, justifications: list[str]) -> GraderResult:
    # Partial credit for correct keywords in justification
    correct_keywords = ["stage 3", "join stage", "schema drift", "SSN", "PII", "type mismatch", "revenue", "aggregation"]
    keyword_bonus = 0.0
    for just in justifications:
        for kw in correct_keywords:
            if kw.lower() in just.lower():
                keyword_bonus += 0.05
    keyword_bonus = min(keyword_bonus, 0.15)
    score = (diagnosis_score * 0.25) + (fix_score * 0.35) + (pii_score * 0.20) + (validation_score * 0.20) + keyword_bonus
    score = max(0.0, min(1.0, score))
    return GraderResult(
        score=round(score, 4),
        breakdown={
            "diagnosis": round(diagnosis_score, 4),
            "fix": round(fix_score, 4),
            "pii_sweep": round(pii_score, 4),
            "validation": round(validation_score, 4),
            "keyword_bonus": round(keyword_bonus, 4)
        },
        explanation=f"Diagnosis: {diagnosis_score}, Fix: {fix_score}, PII: {pii_score}, Validation: {validation_score}, Bonus: {keyword_bonus}"
    )
