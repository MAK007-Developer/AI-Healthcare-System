"""
Backend Report Analysis Module
==============================
Handles the "Smart Lab Analyzer" feature.
Uses Computer Vision to extract numerical data 
from uploaded medical report images (PNG/JPG).

Author: Pavan Badempet
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import logging
from . import vision_service, database, auth, models, pdf_service

# --- Logging ---
# logging.basicConfig(level=logging.INFO) # Handled in main.py
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze/report", response_model=Dict[str, Any])
async def analyze_report(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze an uploaded medical report image.
    
    Args:
        file (UploadFile): Image file (JPEG/PNG).
        
    Returns:
        dict: Extracted metrics and summary.
    
    Raises:
        HTTPException(400): Invalid file type.
        HTTPException(500): Analysis failure.
    """
    # 1. Validate File Type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a JPEG or PNG image."
        )
    
    try:
        # 2. Read File
        contents = await file.read()
        
        # 3. Analyze via Vision Service
        result = vision_service.analyze_lab_report(contents)
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Report Analysis Failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze report")
        raise HTTPException(status_code=500, detail="Failed to analyze report")

# --- PDF Download Endpoint ---
from fastapi import Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

def sanitize_for_pdf(text: Any) -> str:
    """Converts input to a string and filters out characters unsupported by standard PDF fonts."""
    if text is None:
        return "Not Provided"
    text_str = str(text)
    # Encode to latin-1 while ignoring unmappable characters (like emojis), then decode back
    return text_str.encode('latin-1', errors='ignore').decode('latin-1').strip()

@router.get("/download/health-report")
def download_health_report(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    try:
        # 1. Gather Profile Data directly from the User DB record and sanitize immediately
        report_data = {
            "Full Name": sanitize_for_pdf(current_user.full_name),
            "Gender": sanitize_for_pdf(current_user.gender or "Unknown"),
            "Blood Type": sanitize_for_pdf(current_user.blood_type or "Unknown"),
            "Date of Birth": sanitize_for_pdf(current_user.dob),
            "Height": sanitize_for_pdf(f"{current_user.height} cm" if current_user.height else None),
            "Weight": sanitize_for_pdf(f"{current_user.weight} kg" if current_user.weight else None),
            "Diet Type": sanitize_for_pdf(current_user.diet),
            "Activity Level": sanitize_for_pdf(current_user.activity_level),
            "Sleep Hours": sanitize_for_pdf(f"{current_user.sleep_hours} hrs" if current_user.sleep_hours else None),
            "Stress Level": sanitize_for_pdf(current_user.stress_level),
        }
        
        if hasattr(current_user, 'about_me') and current_user.about_me:
            report_data["About Patient"] = sanitize_for_pdf(current_user.about_me)
            
        prediction_val = "General Health Summary"
        advice_list = ["Maintain a balanced diet.", "Engage in regular physical exercise."]

        # 2. Grab the most recent health record if available
        latest_record = db.query(models.HealthRecord).filter(
            models.HealthRecord.user_id == current_user.id
        ).order_by(models.HealthRecord.timestamp.desc()).first()

        if latest_record:
            # Sanitize the prediction string to block emoji crashes
            raw_pred = f"{latest_record.record_type.capitalize()} Analysis: {latest_record.prediction}"
            prediction_val = sanitize_for_pdf(raw_pred)
            
            if latest_record.data:
                import json
                try:
                    if isinstance(latest_record.data, str):
                        parsed_data = json.loads(latest_record.data)
                    else:
                        parsed_data = latest_record.data
                    
                    if isinstance(parsed_data, dict):
                        for k, v in parsed_data.items():
                            clean_key = sanitize_for_pdf(k.replace("_", " ").title())
                            clean_val = sanitize_for_pdf(v)
                            report_data[clean_key] = clean_val
                except Exception as e:
                    logger.warning(f"Failed to parse record data: {e}")

        # Sanitize items in advice list just to be perfectly safe
        clean_advice = [sanitize_for_pdf(tip) for tip in advice_list]

        # 3. Generate PDF safely
        pdf_bytes = pdf_service.generate_medical_report(
            user_name=sanitize_for_pdf(current_user.username),
            report_type="Comprehensive Health Profile",
            prediction=prediction_val,
            data=report_data,
            advice=clean_advice
        )

        return Response(
            content=pdf_bytes, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Health_Report_{current_user.username}.pdf"}
        )

    except Exception as e:
        logger.error(f"PDF Generation Failed: {e}", exc_info=True) # exc_info=True prints full stack trace to backend logs
        raise HTTPException(status_code=500, detail=f"PDF Generation Error: {str(e)}")