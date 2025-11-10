import json
import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from model_handler import llm_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Load model on startup
try:
    llm_handler.load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise


class EmailGeneratorRequest(BaseModel):
    response: str = Field(..., description="Raw Q&A response text to be formatted into email")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "What is Fineanswers?: A: Fineanswers is a platform...\nWho are the contact details?: A: fineanswers.goa@gmail.com"
            }
        }



class EmailFormatter:

    def format_qa_to_email(self, qa_response: str):
        """
        Takes raw Q&A response and formats it into a professional email body.
        Optimized prompt for Phi-3 model.
        """
        
        # Clean up the input
        qa_response = qa_response.strip()
        
        prompt = f"""You are an email writing assistant. Convert the following question-answer pairs into a well-formatted, professional email body.

INSTRUCTIONS:
1. Create a proper email structure with clear formatting
2. Group related questions together if applicable
3. Use bullet points or numbered lists for clarity
4. Remove any "out of context" answers or handle them gracefully
5. Keep the tone professional and courteous
6. Make it easy to read and professional
7. Always include a greeting (Dear Customer, Dear Sir/Madam, or similar)
8. Always include a signature (Best regards, Support Team or similar)

RAW Q&A CONTENT:
{qa_response}

OUTPUT REQUIREMENTS:
- Start with a professional greeting
- Format answers in a clear, structured way
- Use proper paragraph breaks
- End with a professional signature
- Do NOT include subject line
- Output ONLY the email body text

EMAIL BODY:"""

        result = llm_handler.generate(prompt, max_tokens=800, temperature=0.7)
        return self._clean_email_output(result)

    def _clean_email_output(self, raw_output: str) -> str:
        """Clean up LLM output to ensure it's a proper email body."""
        # Remove common artifacts
        cleaned = raw_output.strip()
        
        # Remove "EMAIL BODY:" if model repeats it
        if cleaned.upper().startswith("EMAIL BODY:"):
            cleaned = cleaned[11:].strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        
        return cleaned.strip()


formatter = EmailFormatter()


app = FastAPI(
    title="Email Generator API",
    description="Generate professionally formatted emails from Q&A responses",
    version="1.0.0"
)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Email Generator API",
        "model": "loaded"
    }


@app.post("/generate-email", tags=["Email Generation"])
async def generate_email(request: EmailGeneratorRequest):
    """
    Generate a professionally formatted email body from Q&A response text.
    
    Request body:
    {
        "response": "What is Fineanswers?: A: Fineanswers is..."
    }
    
    Response:
    {
        "status": "success",
        "email_body": "Dear Customer,\\n\\nThank you..."
    }
    """
    try:
        logger.info(f"Generating email from Q&A response")
        
        formatted_email = formatter.format_qa_to_email(qa_response=request.response)
        
        return {
            "status": "success",
            "email_body": formatted_email
        }

    except Exception as e:
        logger.error(f"Email generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down Email Generator API...")
