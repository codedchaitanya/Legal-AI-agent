# api/routes/cases.py
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from db.postgres import AsyncSessionLocal, Case
from api.deps import require_api_key

router = APIRouter()


class CreateCaseRequest(BaseModel):
    title: str
    lawyer_id: str


@router.post("", status_code=201)
async def create_case(req: CreateCaseRequest, _: str = Depends(require_api_key)):
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    async with AsyncSessionLocal() as db:
        db.add(Case(id=case_id, title=req.title, lawyer_id=req.lawyer_id))
        await db.commit()
    return {"case_id": case_id, "title": req.title}


@router.get("/{case_id}")
async def get_case(case_id: str, _: str = Depends(require_api_key)):
    async with AsyncSessionLocal() as db:
        case = await db.get(Case, case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return {"case_id": case.id, "title": case.title, "domains": case.domains}
