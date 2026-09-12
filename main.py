# AegisAI — main.py | Built for SOC & AI Engineers
#
# FastAPI orchestrator: routes requests → Tier-1 Algorithmic → conditional Tier-2 AI (SLM)
# In-memory scan log with telemetry export
# CORS-enabled for React frontend (localhost:5173)

from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from scanner import ScanResult, scan_prompt
from slm_scanner import evaluate_with_gemini
from file_extractor import extract_text

class ScanRequest(BaseModel):
    prompt: str
    session_id: str
    model_output: Optional[str] = None

class ScanLogEntry(BaseModel):
    timestamp: str
    prompt_snippet: str
    risk_score: float
    risk_tier: str
    recommended_action: str
    attack_categories: list[str]
    flagged_keyphrases: list[str]
    tier2_triggered: bool
    attack_type: Optional[str] = None
    canary_detected: bool = False
    vector_similarity: Optional[float] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None

class FileScanResult(BaseModel):
    file_name: str
    file_metadata: dict
    extracted_preview: str    # first 300 chars of extracted text
    scan_result: Optional[ScanResult] = None
    error: Optional[str] = None   # if extraction failed for this file

app = FastAPI(title="AegisAI", version="2.0.0")
scan_log: list[ScanLogEntry] = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _run_tiered_scan(prompt: str, model_output: Optional[str] = None) -> tuple[ScanResult, bool]:
    result = scan_prompt(prompt, model_output)
    tier2_triggered = False

    # Intelligent Tier-2 AI Routing:
    # Trigger AI evaluation if:
    # 1. Risk score is in the ambiguous/warning zone (15 <= score < 70)
    # 2. OR Vector similarity is elevated (>= 0.40) even if heuristics missed
    # 3. OR Anomaly entropy was flagged (> 0.0)
    should_invoke_tier2 = (
        (15.0 <= result.risk_score < 70.0) or
        (result.vector_similarity >= 0.40 and result.risk_score < 70.0) or
        (result.entropy_score > 0.0 and result.risk_score < 70.0)
    )

    if should_invoke_tier2:
        tier2_triggered = True
        gemini_result = await evaluate_with_gemini(
            prompt,
            prompt_hash=result.prompt_hash_for_vector_db
        )

        if gemini_result.get("is_hostile"):
            added_risk = gemini_result.get("added_risk", 35.0)
            result.risk_score = round(min(result.risk_score + added_risk, 100.0), 1)
            result.is_hostile = True
            result.recommended_action = "BLOCK"
            result.risk_tier = "HOSTILE"
            ai_reason = gemini_result.get("reason", "Hostile pattern detected by AI")
            result.flagged_keyphrases.append(f"[AI Detected: {ai_reason}]")
            result.attack_type = gemini_result.get("attack_type")
            if result.attack_type and result.attack_type not in result.attack_categories:
                result.attack_categories.append(result.attack_type)
                result.attack_categories = sorted(result.attack_categories)

    result.tier2_triggered = tier2_triggered
    return result, tier2_triggered

@app.post("/api/scan", response_model=ScanResult)
async def scan_endpoint(request: ScanRequest):
    result, tier2_triggered = await _run_tiered_scan(request.prompt, request.model_output)

    log_entry = ScanLogEntry(
        timestamp=datetime.now().isoformat(),
        prompt_snippet=request.prompt[:80],
        risk_score=result.risk_score,
        risk_tier=result.risk_tier,
        recommended_action=result.recommended_action,
        attack_categories=result.attack_categories,
        flagged_keyphrases=result.flagged_keyphrases,
        tier2_triggered=tier2_triggered,
        attack_type=getattr(result, "attack_type", None),
        canary_detected=result.canary_detected,
        vector_similarity=getattr(result, "vector_similarity", None)
    )
    scan_log.append(log_entry)
    return result

@app.post("/api/scan-files", response_model=List[FileScanResult])
async def scan_files_endpoint(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...)
):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed per scan")

    results: List[FileScanResult] = []
    total_size = 0
    max_total_size = 10 * 1024 * 1024  # 10 MB

    for file in files:
        content = await file.read()
        total_size += len(content)
        if total_size > max_total_size:
            raise HTTPException(status_code=400, detail="Total upload size exceeds 10 MB limit")

        try:
            extracted_text, file_meta = extract_text(content, file.filename or "unknown")
            scan_res, tier2_trig = await _run_tiered_scan(extracted_text)

            log_entry = ScanLogEntry(
                timestamp=datetime.now().isoformat(),
                prompt_snippet=f"{file.filename}: {extracted_text[:60]}",
                risk_score=scan_res.risk_score,
                risk_tier=scan_res.risk_tier,
                recommended_action=scan_res.recommended_action,
                attack_categories=scan_res.attack_categories,
                flagged_keyphrases=scan_res.flagged_keyphrases,
                tier2_triggered=tier2_trig,
                attack_type=getattr(scan_res, "attack_type", None),
                canary_detected=scan_res.canary_detected,
                vector_similarity=getattr(scan_res, "vector_similarity", None),
                file_name=file.filename,
                file_type=file_meta.get("type")
            )
            scan_log.append(log_entry)

            results.append(FileScanResult(
                file_name=file.filename or "unknown",
                file_metadata=file_meta,
                extracted_preview=extracted_text[:300],
                scan_result=scan_res
            ))
        except HTTPException as he:
            results.append(FileScanResult(
                file_name=file.filename or "unknown",
                file_metadata={},
                extracted_preview="",
                scan_result=None,
                error=he.detail
            ))
        except Exception as e:
            results.append(FileScanResult(
                file_name=file.filename or "unknown",
                file_metadata={},
                extracted_preview="",
                scan_result=None,
                error=str(e)
            ))

    return results

@app.post("/api/scan-file", response_model=FileScanResult)
async def scan_file_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    res = await scan_files_endpoint(files=[file], session_id=session_id)
    return res[0]

@app.get("/api/telemetry")
def get_telemetry():
    return {"scans": list(reversed(scan_log[-20:]))}

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "total_scans": len(scan_log),
        "timestamp": datetime.now().isoformat()
    }

# Connect Frontend static bundle for unified deployment
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        target = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(target):
            return FileResponse(target)
        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
        return {"error": "Frontend build not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
