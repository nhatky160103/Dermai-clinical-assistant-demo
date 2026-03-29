"""
🔬 DermAI Clinical Assistant — Main Application

RAG-based Clinical Decision Support Pipeline for Dermatology
Combines Task 2 (Segmentation) + Task 3 (Classification) + RAG + LLM

Usage:
    python app.py
    # Then open http://localhost:7860 in your browser

Environment variables:
    LLM_PROVIDER               — LLM backend selector
    GEMINI_API_KEY             — Gemini API key for REST/OpenAI-compatible access
    GEMINI_MODEL               — Gemini model name
    TASK2_MODEL_PATH           — Path to Task 2 model weights
    TASK2_MODEL_TYPE           — "transunet" for notebook-compatible Task 2 loading
    TASK2_TRANSUNET_SOURCE_DIR — Root directory containing TransUNet `networks/`
    TASK2_TRANSUNET_IMG_SIZE   — Task 2 inference image size
    TASK2_TRANSUNET_VIT_NAME   — Task 2 ViT backbone name
    TASK3_MODEL_PATH           — Path to Task 3 model weights
    TASK3_MODEL_TYPE           — "isic" for repository classifier loader
    TASK3_BACKBONE             — Task 3 backbone name
"""
import os
import sys
import time
import traceback
import numpy as np
from PIL import Image

import gradio as gr

# ─── Imports ───
from config import (
    APP_TITLE, APP_DESCRIPTION,
    ATTRIBUTE_LABELS, ATTRIBUTE_COLORS, ATTRIBUTE_DESCRIPTIONS,
    DISEASE_FULL_NAMES, DISEASE_COLORS, DISEASE_RISK_LEVELS,
    TASK2_MODEL_PATH, TASK3_MODEL_PATH,
)
from models.task2_segmenter.task2_segmenter import get_segmenter
from models.task3_classifier.task3_classifier import get_classifier
from pipeline.context_builder import build_context
from pipeline.image_search import search_similar_cases, format_similar_cases_text, initialize_reference_database
from pipeline.llm_explainer import get_llm_explanation
from pipeline.report_generator import (
    create_annotated_image,
    create_structure_heatmap,
    create_diagnosis_chart,
    generate_report_markdown,
)


# ─── Initialize models (lazy, on first call) ───
segmenter = None
classifier = None


def init_models():
    global segmenter, classifier
    if segmenter is None:
        segmenter = get_segmenter(TASK2_MODEL_PATH)
        print(f"✓ Task 2 Segmenter: {segmenter.model_name}")
    if classifier is None:
        classifier = get_classifier(TASK3_MODEL_PATH)
        print(f"✓ Task 3 Classifier: {classifier.model_name}")


# ──────────────────────────────────────────────────────────────── Main Pipeline
def analyze_image(image, patient_age, patient_sex, lesion_location):
    """
    Main pipeline function — orchestrates the full clinical analysis.

    Args:
        image: Uploaded PIL Image
        patient_age: Optional patient age
        patient_sex: Optional patient sex
        lesion_location: Optional lesion location

    Returns:
        Tuple of Gradio outputs for each tab
    """
    if image is None:
        raise gr.Error("Please upload a dermoscopy image first.")

    try:
        init_models()
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        # ─── Step 1: Task 2 — Segmentation ───
        seg_results = segmenter.predict(pil_image)

        # ─── Step 2: Task 3 — Classification ───
        cls_result = classifier.predict(pil_image, top_k=3)

        # ─── Step 3: Build Context ───
        patient_meta = {}
        if patient_age:
            patient_meta["Age"] = patient_age
        if patient_sex:
            patient_meta["Sex"] = patient_sex
        if lesion_location:
            patient_meta["Lesion Location"] = lesion_location

        context = build_context(
            segmentation_results=seg_results,
            classification_result=cls_result,
            patient_metadata=patient_meta if patient_meta else None,
        )

        # ─── Step 4: RAG — Similar Cases ───
        similar_cases = search_similar_cases(pil_image, context, top_k=3)
        similar_text = format_similar_cases_text(similar_cases)

        # ─── Step 5: LLM Explanation ───
        llm_explanation = get_llm_explanation(context, similar_text)

        # ─── Step 6: Generate Visualizations ───
        annotated_img = create_annotated_image(pil_image, seg_results)
        heatmap_img = create_structure_heatmap(pil_image, seg_results)
        diagnosis_chart = create_diagnosis_chart(cls_result)

        # ─── Step 7: Generate Report ───
        report_md = generate_report_markdown(context, llm_explanation, similar_cases)

        # ─── Format structure analysis table (HTML) ───
        structure_html = format_structure_table(seg_results, context)

        # ─── Format diagnosis output ───
        diagnosis_html = format_diagnosis_output(cls_result, context)

        # ─── Format similar cases ───
        cases_html = format_similar_cases_html(similar_cases)

        # ─── Risk banner ───
        risk_html = format_risk_banner(context)

        return (
            risk_html,          # Risk banner
            annotated_img,      # Annotated image
            heatmap_img,        # Structure heatmap
            structure_html,     # Structure table
            diagnosis_chart,    # Diagnosis chart
            diagnosis_html,     # Diagnosis details
            cases_html,         # Similar cases
            llm_explanation,    # LLM explanation
            report_md,          # Full report
        )

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Analysis failed: {str(e)}")


# ──────────────────────────────────────────────────────────────── Formatters

def format_risk_banner(context) -> str:
    """Generate HTML risk banner based on clinical context."""
    if context.risk_level == "HIGH":
        cls = "risk-high"
        icon = "🔴"
        msg = f"HIGH RISK — Primary diagnosis: {context.primary_diagnosis_name} ({context.primary_confidence:.0%}). Urgent review recommended."
    elif context.risk_level == "MODERATE":
        cls = "risk-moderate"
        icon = "🟡"
        msg = f"MODERATE RISK — Primary diagnosis: {context.primary_diagnosis_name} ({context.primary_confidence:.0%}). Clinical correlation recommended."
    else:
        cls = "risk-low"
        icon = "🟢"
        msg = f"LOW RISK — Primary diagnosis: {context.primary_diagnosis_name} ({context.primary_confidence:.0%}). Routine monitoring."

    if context.is_uncertain:
        msg += " ⚠️ Model uncertainty is high."

    return f'<div class="risk-banner {cls}">{icon} {msg}</div>'


def format_structure_table(seg_results: dict, context) -> str:
    """Generate HTML table for detected structures."""
    rows = []
    for attr, result in seg_results.items():
        color = ATTRIBUTE_COLORS.get(attr, (150, 150, 150))
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        status = "✅ Detected" if result.present else "— Not detected"
        conf_pct = f"{result.confidence:.0%}"
        cov = f"{result.coverage_pct:.1f}%" if result.present else "—"
        desc = ATTRIBUTE_DESCRIPTIONS.get(attr, "")

        opacity = "1" if result.present else "0.4"
        rows.append(f"""
        <tr style="opacity: {opacity}">
            <td><span style="color: {color_hex}; font-weight: 600;">●</span> {ATTRIBUTE_LABELS.get(attr, attr)}</td>
            <td>{status}</td>
            <td>{conf_pct}</td>
            <td>{cov}</td>
            <td style="font-size: 0.85em; color: #9ba3b5;">{desc}</td>
        </tr>""")

    return f"""
    <table class="structure-table">
        <thead>
            <tr>
                <th>Structure</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Coverage</th>
                <th>Clinical Significance</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    <p style="margin-top: 0.75rem; font-size: 0.85rem; color: #636d83;">
        {context.total_structures_detected}/5 structures detected • 
        Coverage = % of image area occupied by structure
    </p>
    """


def format_diagnosis_output(cls_result, context) -> str:
    """Generate HTML for diagnosis details."""
    parts = []

    # Top diagnosis
    primary = cls_result.top_k[0]
    risk_color = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#10b981"}.get(primary.risk_level, "#9ba3b5")

    parts.append(f"""
    <div style="background: rgba(79, 140, 255, 0.06); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
        <div style="font-size: 0.85rem; color: #9ba3b5; margin-bottom: 0.25rem;">PRIMARY AI DIAGNOSIS</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #e8eaf0;">
            {primary.name}
            <span style="font-size: 0.9rem; font-weight: 500; color: {risk_color}; 
                         background: {risk_color}20; padding: 0.2rem 0.6rem; border-radius: 20px; margin-left: 0.5rem;">
                {primary.risk_level} RISK
            </span>
        </div>
        <div style="font-size: 1.1rem; color: #4f8cff; font-weight: 600; margin-top: 0.25rem;">
            {primary.confidence:.1%} confidence
        </div>
    </div>
    """)

    if cls_result.is_uncertain:
        parts.append("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); 
                    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; color: #fcd34d; font-weight: 500;">
            ⚠️ Model uncertainty is HIGH — Top diagnosis confidence &lt; 50%. 
            Clinical correlation strongly recommended.
        </div>
        """)

    # Differential diagnosis table
    parts.append("""
    <div style="font-weight: 600; margin: 1rem 0 0.5rem; color: #e8eaf0;">Differential Diagnosis</div>
    <table class="structure-table">
        <thead><tr><th>Rank</th><th>Diagnosis</th><th>Confidence</th><th>Risk</th></tr></thead>
        <tbody>
    """)

    for i, d in enumerate(cls_result.top_k, 1):
        risk_badge_color = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#10b981"}.get(d.risk_level, "#9ba3b5")
        parts.append(f"""
        <tr>
            <td>#{i}</td>
            <td><strong>{d.name}</strong> ({d.code})</td>
            <td>{d.confidence:.1%}</td>
            <td><span style="color: {risk_badge_color}; font-weight: 600;">{d.risk_level}</span></td>
        </tr>""")

    parts.append("</tbody></table>")

    # Risk flags
    if context.risk_flags:
        parts.append('<div style="margin-top: 1rem;">')
        for flag in context.risk_flags:
            parts.append(f'<div style="padding: 0.4rem 0; color: #fca5a5; font-size: 0.9rem;">{flag}</div>')
        parts.append("</div>")

    return "".join(parts)


def format_similar_cases_html(cases) -> str:
    """Generate HTML cards for similar reference cases."""
    if not cases:
        return '<p style="color: #636d83;">No similar reference cases found in the knowledge base.</p>'

    parts = [f'<div style="font-weight: 600; color: #e8eaf0; margin-bottom: 1rem;">Found {len(cases)} similar reference cases:</div>']

    for i, case in enumerate(cases, 1):
        risk_color = {"HIGH": "#ef4444", "MODERATE": "#f59e0b"}.get(
            DISEASE_RISK_LEVELS.get(case.diagnosis, "LOW"), "#10b981"
        )
        structures = ", ".join(s.replace("_", " ").title() for s in case.structures) if case.structures else "None"
        evidence = (
            f"{getattr(case, 'diagnosis_confirm_type', 'single image expert consensus')} "
            f"(w={getattr(case, 'evidence_weight', 0.6):.2f})"
        )

        parts.append(f"""
        <div class="case-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; color: #e8eaf0;">Reference Case {i}</span>
                <span style="font-size: 0.85rem; color: #4f8cff;">Similarity: {case.similarity_score:.0%}</span>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <span class="diagnosis-badge" style="border-color: {risk_color}30; color: {risk_color}; background: {risk_color}15;">
                    {case.diagnosis_name} ({case.diagnosis})
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #9ba3b5; margin-bottom: 0.25rem;">
                <strong>Structures:</strong> {structures}
            </div>
            <div style="font-size: 0.85rem; color: #9ba3b5; margin-bottom: 0.25rem;">
                <strong>Evidence:</strong> {evidence}
            </div>
            <div style="font-size: 0.85rem; color: #9ba3b5; line-height: 1.5;">
                {case.description[:300]}{'...' if len(case.description) > 300 else ''}
            </div>
            <div style="font-size: 0.8rem; color: #636d83; margin-top: 0.5rem;">Source: {case.source}</div>
        </div>
        """)

    return "".join(parts)


# ──────────────────────────────────────────────────────────────── Gradio UI

def create_ui():
    """Build the Gradio Blocks interface."""

    css_path = os.path.join(os.path.dirname(__file__), "ui", "styles.css")
    with open(css_path, "r") as f:
        custom_css = f.read()

    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#eef2ff", c100="#e0e7ff", c200="#c7d2fe",
            c300="#a5b4fc", c400="#818cf8", c500="#6366f1",
            c600="#4f46e5", c700="#4338ca", c800="#3730a3",
            c900="#312e81", c950="#1e1b4b",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0",
            c300="#cbd5e1", c400="#94a3b8", c500="#64748b",
            c600="#475569", c700="#334155", c800="#1e293b",
            c900="#0f172a", c950="#020617",
        ),
    )

    with gr.Blocks(title="DermAI Clinical Assistant") as demo:
        # ─── Header ───
        gr.HTML(f"""
        <div class="app-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_DESCRIPTION}</p>
            <p style="font-size: 0.8rem; color: #636d83; margin-top: 0.5rem;">
                Powered by TransUNet (Task 2) • EfficientNet (Task 3) • CLIP • ChromaDB RAG • LLM
            </p>
        </div>
        """)

        with gr.Row():
            # ─── Left: Input Panel ───
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### 📷 Upload Dermoscopy Image")
                input_image = gr.Image(
                    type="pil",
                    label="Dermoscopy Image",
                    height=350,
                    elem_classes=["upload-area"],
                )

                gr.Markdown("### 👤 Patient Information *(optional)*")
                with gr.Row():
                    patient_age = gr.Textbox(
                        label="Age",
                        placeholder="e.g., 45",
                        max_lines=1,
                    )
                    patient_sex = gr.Dropdown(
                        label="Sex",
                        choices=["", "Male", "Female", "Other"],
                        value="",
                    )
                lesion_location = gr.Textbox(
                    label="Lesion Location",
                    placeholder="e.g., left forearm, upper back",
                    max_lines=1,
                )

                analyze_btn = gr.Button(
                    "🔬 Analyze Image",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("""
                <div style="font-size: 0.8rem; color: #636d83; margin-top: 1rem; line-height: 1.5;">
                    <strong>How it works:</strong><br>
                    1️⃣ Upload a dermoscopy image<br>
                    2️⃣ AI detects dermoscopic structures (Task 2)<br>
                    3️⃣ AI classifies the disease (Task 3)<br>
                    4️⃣ RAG retrieves similar reference cases<br>
                    5️⃣ LLM generates clinical interpretation<br>
                    6️⃣ Full clinical report is assembled
                </div>
                """)

            # ─── Right: Results ───
            with gr.Column(scale=2):
                # Risk banner
                risk_banner = gr.HTML(
                    value='<div style="color: #636d83; text-align: center; padding: 2rem;">Upload an image to start analysis</div>',
                )

                with gr.Tabs() as tabs:
                    # Tab 1: Structure Analysis
                    with gr.Tab("🔍 Structure Analysis", id="structures"):
                        annotated_image = gr.Image(
                            label="Annotated Image (Segmentation Overlay)",
                            type="pil",
                            height=400,
                        )
                        heatmap_image = gr.Image(
                            label="Individual Structure Heatmaps",
                            type="pil",
                            height=250,
                        )
                        structure_table = gr.HTML(label="Detected Structures")

                    # Tab 2: Diagnosis
                    with gr.Tab("🎯 Diagnosis", id="diagnosis"):
                        diagnosis_chart = gr.Image(
                            label="Confidence Distribution",
                            type="pil",
                            height=350,
                        )
                        diagnosis_details = gr.HTML(label="Diagnosis Details")

                    # Tab 3: Similar Cases (RAG)
                    with gr.Tab("📚 Similar Cases (RAG)", id="cases"):
                        similar_cases_html = gr.HTML(label="Reference Cases")

                    # Tab 4: AI Explanation (LLM)
                    with gr.Tab("🤖 AI Interpretation", id="llm"):
                        llm_output = gr.Markdown(
                            label="Clinical Interpretation",
                            elem_classes=["report-content"],
                        )

                    # Tab 5: Full Report
                    with gr.Tab("📋 Full Report", id="report"):
                        full_report = gr.Markdown(
                            label="Clinical Report",
                            elem_classes=["report-content"],
                        )

        # ─── Wire up the analysis ───
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, patient_age, patient_sex, lesion_location],
            outputs=[
                risk_banner,
                annotated_image,
                heatmap_image,
                structure_table,
                diagnosis_chart,
                diagnosis_details,
                similar_cases_html,
                llm_output,
                full_report,
            ],
        )

        # ─── Footer ───
        gr.HTML("""
        <div style="text-align: center; padding: 1.5rem; margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.06);">
            <p style="font-size: 0.85rem; color: #636d83;">
                ⚕️ <strong>Disclaimer:</strong> This is an AI-assisted clinical decision support tool. 
                It does not constitute a medical diagnosis. All findings must be correlated with 
                clinical examination and professional medical judgment.
            </p>
            <p style="font-size: 0.75rem; color: #4a5568; margin-top: 0.5rem;">
                ISIC 2018 Challenge • TransUNet + EfficientNet + CLIP + ChromaDB + LLM Pipeline
            </p>
        </div>
        """)

    return demo, custom_theme, custom_css


# ──────────────────────────────────────────────────────────────── Main
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {APP_TITLE}")
    print("=" * 60)

    # Initialize reference database for RAG
    try:
        initialize_reference_database()
    except Exception as e:
        print(f"⚠ Reference database init warning: {e}")
        print("  RAG will use fallback keyword-based search.")

    # Initialize models
    init_models()

    # Launch
    demo, custom_theme, custom_css = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=custom_theme,
        css=custom_css,
    )
