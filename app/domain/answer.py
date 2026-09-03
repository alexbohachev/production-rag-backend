from app.domain.ranking import Chunk
from app.schemas import Citation
from app.textutil import tokenize


def extractive_answer(query: str, ranked: list[Chunk]) -> tuple[str, list[Citation], float]:
    if not ranked:
        return "No supporting documents found.", [], 0.0

    q = set(tokenize(query))
    citations: list[Citation] = []
    for i, chunk in enumerate(ranked):
        overlap = len(q & set(tokenize(chunk.text))) / max(len(q), 1)
        quote = chunk.text if len(chunk.text) <= 220 else chunk.text[:217] + "..."
        citations.append(
            Citation(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                quote=quote,
                score=round(max(overlap, 0.15) * (1.0 - i * 0.05), 4),
            )
        )

    lead = ranked[0]
    answer = f"{lead.title}: {lead.text}"
    if len(ranked) > 1:
        answer += f" Related: {ranked[1].title}."
    confidence = min(0.95, 0.45 + citations[0].score)
    return answer, citations, round(confidence, 3)
