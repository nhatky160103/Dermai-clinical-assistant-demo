"""
LLM Clinical Explainer — AI Agent that generates clinical interpretations.

Uses OpenAI-compatible API (GPT, Claude, Ollama) to translate model outputs
into clinical language that dermatologists can understand and verify.
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    VERTEX_API_KEY,
    VERTEX_BASE_URL,
    LLM_MODEL,
    VERTEX_MODEL,
)


# ──────────────────────────────────────────────────────── System Prompt
SYSTEM_PROMPT = """You are a clinical dermatology AI assistant specializing in dermoscopy interpretation.

Your role is to:
1. Interpret AI-detected dermoscopic structures and their clinical significance
2. Explain why detected structures support or contradict the AI diagnosis
3. Highlight key features the dermatologist should verify manually
4. Recommend appropriate next steps based on findings

IMPORTANT GUIDELINES:
- You are providing decision SUPPORT, not making diagnoses
- Always include uncertainty and recommend clinical correlation
- Use standard dermatological terminology
- Be structured and concise — clinicians are time-constrained
- When findings are alarming (melanoma, BCC), be clear but not alarmist
- Reference the ABCDE criteria and 7-point checklist when relevant
- Always end with a disclaimer about AI-assisted analysis

Format your response with clear sections:
1. **Clinical Significance of Detected Structures**
2. **Diagnostic Assessment**  
3. **Key Features to Verify**
4. **Recommended Next Steps**
5. **AI Confidence & Limitations**"""


def build_clinical_prompt(clinical_context, similar_cases_text: str = "") -> str:
    """
    Build the structured prompt for LLM clinical explanation.

    Args:
        clinical_context: ClinicalContext object
        similar_cases_text: Formatted text of similar reference cases

    Returns:
        Complete prompt string
    """
    prompt_parts = [
        "Please provide a clinical interpretation of the following dermoscopy analysis.",
        "",
        clinical_context.context_text,
    ]

    if similar_cases_text:
        prompt_parts += [
            "",
            "=== SIMILAR REFERENCE CASES (from knowledge base) ===",
            similar_cases_text,
        ]

    prompt_parts += [
        "",
        "Based on the above AI analysis and reference cases, please provide:",
        "1. Clinical significance of each detected dermoscopic structure",
        "2. Whether the detected structures support or contradict the AI diagnosis",
        "3. Specific features the dermatologist should verify by examining the image",
        "4. Recommended next steps (biopsy, monitoring, reassurance, etc.)",
        "5. Any limitations or uncertainties in this AI-assisted analysis",
        "",
        "Remember: This is for a clinical decision support tool. Be precise, structured, and always recommend clinical correlation.",
    ]

    return "\n".join(prompt_parts)


def get_llm_explanation(
    clinical_context,
    similar_cases_text: str = "",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Get clinical explanation from LLM.

    Args:
        clinical_context: ClinicalContext object
        similar_cases_text: Formatted text of similar reference cases
        api_key: OpenAI API key (falls back to env var)
        model: Model name (falls back to config)

    Returns:
        Clinical explanation text
    """
    # Provider selection:
    # - Explicit: LLM_PROVIDER=vertex
    # - Implicit: no OpenAI key but has VERTEX_API_KEY
    use_vertex = LLM_PROVIDER == "vertex" or (not OPENAI_API_KEY and bool(VERTEX_API_KEY))

    if use_vertex:
        key = api_key or VERTEX_API_KEY
        base_url = VERTEX_BASE_URL
        mdl = model or VERTEX_MODEL
    else:
        key = api_key or OPENAI_API_KEY
        base_url = OPENAI_BASE_URL
        mdl = model or LLM_MODEL

    if not key:
        return _generate_fallback_explanation(clinical_context)

    prompt = build_clinical_prompt(clinical_context, similar_cases_text)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=key,
            base_url=base_url,
        )

        response = client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    except ImportError:
        print("⚠ OpenAI package not installed. Using fallback explanation.")
        return _generate_fallback_explanation(clinical_context)
    except Exception as e:
        print(f"⚠ LLM API error: {e}. Using fallback explanation.")
        return _generate_fallback_explanation(clinical_context)


def _generate_fallback_explanation(clinical_context) -> str:
    """Generate a rule-based clinical explanation when LLM is unavailable."""

    ctx = clinical_context
    parts = []

    # ─── 1. Structure significance ───
    parts.append("## 1. Clinical Significance of Detected Structures\n")
    if ctx.detected_structures:
        for s in ctx.detected_structures:
            parts.append(f"**{s.label}** (coverage: {s.coverage_pct:.1f}%, confidence: {s.confidence:.0%})")
            parts.append(f"- {s.clinical_description}")

            # Add clinical interpretation
            if s.attribute == "pigment_network":
                parts.append("- Indicates melanocytic origin. Regular network suggests benignity; irregular/atypical network raises concern for melanoma.")
            elif s.attribute == "negative_network":
                parts.append("- Inverse pattern associated with specific melanoma subtypes (Spitzoid, BAP1-mutated). Also seen in dermatofibromas.")
            elif s.attribute == "streaks":
                parts.append("- Radial projections at lesion periphery suggest active growth. In an asymmetric distribution, strongly associated with melanoma.")
            elif s.attribute == "milia_like_cyst":
                parts.append("- White-yellow globules typically associated with benign keratoses (seborrheic keratosis). Reassuring finding.")
            elif s.attribute == "globules":
                parts.append("- Correspond to melanocyte nests. Regular pattern = benign; irregular distribution = concerning.")
            parts.append("")
    else:
        parts.append("No dermoscopic structures were confidently detected. This may indicate:\n- Non-melanocytic lesion\n- Featureless lesion requiring clinical correlation\n- Technical limitations in image quality\n")

    # ─── 2. Diagnostic assessment ───
    parts.append("## 2. Diagnostic Assessment\n")
    parts.append(f"**Primary AI Diagnosis: {ctx.primary_diagnosis_name} ({ctx.primary_diagnosis})** — {ctx.primary_confidence:.0%} confidence\n")

    if ctx.is_uncertain:
        parts.append("> ⚠️ **UNCERTAIN**: The AI model's top prediction has less than 50% confidence. This case requires careful clinical evaluation.\n")

    parts.append("**Differential Diagnosis:**")
    for d in ctx.top_3_diagnoses:
        indicator = "🔴" if d["risk_level"] == "HIGH" else "🟡" if d["risk_level"] == "MODERATE" else "🟢"
        parts.append(f"- {indicator} {d['name']} ({d['code']}): {d['confidence']:.1%}")
    parts.append("")

    # ─── 3. Features to verify ───
    parts.append("## 3. Key Features to Verify\n")
    parts.append("The following features should be verified by the examining dermatologist:")
    parts.append("- Asymmetry of structure and colors (ABCDE criterion A)")
    parts.append("- Border regularity and sharpness (ABCDE criterion B)")
    parts.append("- Color homogeneity vs. multiple colors (ABCDE criterion C)")
    if ctx.risk_level == "HIGH":
        parts.append("- **Urgent**: Confirm presence/absence of regression structures")
        parts.append("- **Urgent**: Evaluate for blue-white veil or atypical vascular pattern")
    parts.append("")

    # ─── 4. Next steps ───
    parts.append("## 4. Recommended Next Steps\n")
    if ctx.risk_level == "HIGH":
        parts.append("🔴 **HIGH RISK — Immediate Action Recommended:**")
        parts.append("- Excisional biopsy with appropriate margins")
        parts.append("- Dermatopathology evaluation with immunohistochemistry if needed")
        parts.append("- Patient follow-up within 1-2 weeks for results")
    elif ctx.risk_level == "MODERATE":
        parts.append("🟡 **MODERATE RISK — Close Monitoring:**")
        parts.append("- Consider biopsy if clinical suspicion is high")
        parts.append("- Short-term digital dermoscopic follow-up (3 months)")
        parts.append("- Document with clinical photography")
    else:
        parts.append("🟢 **LOW RISK — Standard Management:**")
        parts.append("- Routine monitoring as per standard guidelines")
        parts.append("- Patient education on ABCDE self-screening")
        parts.append("- Re-evaluate if changes are reported")
    parts.append("")

    # ─── 5. Limitations ───
    parts.append("## 5. AI Confidence & Limitations\n")
    parts.append(f"- Model uncertainty: {'HIGH' if ctx.is_uncertain else 'Within acceptable range'}")
    parts.append(f"- Number of structures detected: {ctx.total_structures_detected}/5")
    if ctx.risk_flags:
        parts.append("- **Risk flags identified:**")
        for flag in ctx.risk_flags:
            parts.append(f"  - {flag}")
    parts.append("")
    parts.append("> ⚕️ **Disclaimer**: This is an AI-assisted analysis tool for clinical decision support only. "
                 "It does not constitute a medical diagnosis. All findings must be correlated with clinical "
                 "examination, patient history, and professional medical judgment. The treating physician "
                 "is solely responsible for the final diagnostic and therapeutic decisions.")

    return "\n".join(parts)
