import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# =====================================================================
# CONFIG PAGE
# =====================================================================
st.set_page_config(
    page_title="SenClinique AI Pro",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# CSS MODERNE — DESIGN MÉDICAL 2025
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #f8fafc;
    color: #0f172a;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1450px;
}

/* Header */
.app-header {
    background: white;
    padding: 1.2rem 2.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.app-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #0ea5e9, #14b8a6);
    border-radius: 12px;
    color: white;
    font-size: 1.6rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}

.app-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1e293b;
}

.app-subtitle {
    font-size: 0.95rem;
    color: #64748b;
}

/* Cartes */
.card {
    background: transparent !important;
    border-radius: 16px;
    border: none !important;
    box-shadow: none !important;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
}

/* Résultat principal */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.result-normal {
    background: linear-gradient(135deg, #f0fdf4, #ffffff);
    border: 2px solid #4ade80;
}

.result-alert {
    background: linear-gradient(135deg, #fff7ed, #ffffff);
    border: 2px solid #fb923c;
}

.result-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 1rem;
}

.result-normal .result-badge { background: #86efac; color: #166534; }
.result-alert .result-badge  { background: #fed7aa; color: #9a3412; }

.result-confidence {
    font-size: 4.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 1rem 0;
}

.result-normal .result-confidence { color: #15803d; }
.result-alert .result-confidence  { color: #c2410c; }

/* Metrics */
.metric-container {
    display: flex;
    gap: 1.5rem;
    margin: 1.5rem 0;
}

.metric-item {
    flex: 1;
    background: white;
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
}

.metric-label {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Recommandations */
.reco-item {
    background: #f8fafc;
    padding: 1rem 1.25rem;
    border-radius: 10px;
    border-left: 4px solid #0ea5e9;
    margin-bottom: 0.75rem;
}

/* Bouton PDF */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0ea5e9, #14b8a6) !important;
    color: white !important;
    border: none !important;
    padding: 14px 28px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    width: 100% !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.3) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
}

/* ====================== UPLOADER FIX (Version lisible) ====================== */
.stFileUploader > div > div {
    background: transparent !important;
    border: 2px dashed #94a3b8 !important;
    border-radius: 12px !important;
    padding: 2.5rem 1rem !important;
}

.stFileUploader label {
    color: #64748b !important;
}

/* Bouton "Browse files" - Fond bleu foncé + texte blanc bien visible */
.stFileUploader > div > div > button {
    background: #1e40af !important;        /* Bleu foncé contrasté */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

.stFileUploader button span,
.stFileUploader button div,
.stFileUploader button p {
    color: white !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.6) !important;
}

/* Masquer éléments natifs */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HEADER (inchangé)
# =====================================================================
now = datetime.now()
date_str = now.strftime("%d %B %Y - %H:%M")

st.markdown(f"""
<div class="app-header">
    <div style="display:flex; align-items:center; gap:16px;">
        <div class="app-logo">🫁</div>
        <div>
            <div class="app-title">SenClinique AI Pro</div>
            <div class="app-subtitle">Diagnostic assisté par IA — Pneumonie</div>
        </div>
    </div>
    <div style="text-align:right; color:#64748b;">
        Nom du médecin<br>
        <span style="font-size:0.9rem;">Modèle Hybride CNN+ViT (M5)</span><br>
        {date_str}
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# MODÈLE (inchangé)
# =====================================================================
class HybrideCNNViT(nn.Module):
    def __init__(self, nb_classes=2, d_model=512, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Linear(512, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, nb_classes))

    def forward(self, x):
        B = x.size(0)
        f = self.backbone(x).flatten(2).transpose(1, 2)
        f = self.proj(f)
        c = self.cls_token.expand(B, -1, -1)
        f = torch.cat([c, f], dim=1) + self.pos_embed
        f = self.transformer(f)
        return self.classifier(self.norm(f)[:, 0])

@st.cache_resource
def charger_modele():
    model = HybrideCNNViT()
    model.load_state_dict(torch.load("models/model5_cnn_vit_finetuned_optuna.pth", map_location="cpu"))
    model.eval()
    return model

def pretraiter(image):
    img = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (224, 224))
    img = np.stack([img, img, img], axis=-1)
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t(img).unsqueeze(0)

# =====================================================================
# PDF (inchangé)
# =====================================================================
def generer_pdf(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                            topMargin=25*mm, bottomMargin=20*mm, 
                            leftMargin=25*mm, rightMargin=25*mm)
    
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle("Title", parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER, spaceAfter=20)
    s_section = ParagraphStyle("Section", parent=styles['Heading2'], fontSize=14, spaceBefore=18, spaceAfter=10)
    s_body = ParagraphStyle("Body", fontSize=11, leading=14, spaceAfter=8)
    
    story = []
    
    story.append(Paragraph("RAPPORT DE DIAGNOSTIC IA — SENCLINIQUE", s_title))
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", 
                          ParagraphStyle("Date", fontSize=10, alignment=TA_CENTER, spaceAfter=30)))
    
    story.append(Paragraph("1. Informations Patient", s_section))
    info_data = [
        ["Nom du patient", data['nom']],
        ["Âge / Sexe", f"{data['age']} ans / {data['sexe']}"],
        ["ID Examen", data['id']],
        ["Date d'analyse", datetime.now().strftime("%d/%m/%Y")]
    ]
    t_info = Table(info_data, colWidths=[60*mm, 110*mm])
    t_info.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 12*mm))
    
    story.append(Paragraph("2. Résultat de l'Analyse IA", s_section))
    
    couleur = colors.green if data['res'] == "NORMAL" else colors.red
    res_text = f"{data['res']} — Confiance : {data['conf']:.1f}%"
    
    story.append(Paragraph(res_text, ParagraphStyle("Result", fontSize=16, textColor=couleur, alignment=TA_CENTER, spaceAfter=15)))
    story.append(Spacer(1, 8*mm))
    
    story.append(Paragraph("3. Recommandations Cliniques", s_section))
    for reco in data['recos']:
        story.append(Paragraph(f"• {reco}", s_body))
    
    story.append(Spacer(1, 25*mm))
    story.append(Paragraph(
        "Avertissement : Ce rapport est généré par intelligence artificielle à titre d'aide à la décision. "
        "Il ne remplace pas l'avis d'un médecin spécialiste. Toute décision clinique relève de la responsabilité du professionnel de santé.",
        ParagraphStyle("Warning", fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# =====================================================================
# INTERFACE PRINCIPALE (inchangée)
# =====================================================================
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Données Patient</div>', unsafe_allow_html=True)
    
    nom_patient = st.text_input("Nom complet", placeholder="Ex : Etienne Bledou")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Âge", 0, 120, 45)
    with c2:
        sexe = st.selectbox("Sexe", ["Homme", "Femme", "Non renseigné"])
    
    st.markdown('<div class="card-title" style="margin-top:1.5rem;">Radiographie Thoracique</div>', unsafe_allow_html=True)
    fichier = st.file_uploader("Importer l'image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    
    if fichier:
        image = Image.open(fichier)
        st.image(image, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Analyse IA en Temps Réel</div>', unsafe_allow_html=True)
    
    if fichier is None:
        st.info("Chargez une radiographie thoracique pour lancer l'analyse.")
    else:
        with st.spinner("Analyse des motifs pulmonaires par le modèle Hybride CNN+ViT..."):
            model = charger_modele()
            input_tensor = pretraiter(image)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                p_pneumo = probs[1].item()
                p_normal = probs[0].item()
        
        if p_pneumo > p_normal:
            prediction = "PNEUMONIE"
            confiance = p_pneumo * 100
            classe_css = "result-alert"
        else:
            prediction = "NORMAL"
            confiance = p_normal * 100
            classe_css = "result-normal"
        
        st.markdown(f"""
        <div class="result-card {classe_css}">
            <div class="result-badge">{prediction}</div>
            <div class="result-confidence">{confiance:.1f}%</div>
            <div style="color:#64748b; font-size:1rem;">Confiance du modèle</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value" style="color:#4ade80;">{p_normal*100:.1f}%</div>
                <div class="metric-label">Normal</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#fb923c;">{p_pneumo*100:.1f}%</div>
                <div class="metric-label">Pneumonie</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if prediction == "NORMAL":
            recommandations = [
                "Aucune anomalie pulmonaire majeure détectée.",
                "Suivi clinique standard recommandé.",
                "Ce résultat IA doit être confirmé par un radiologue.",
            ]
        else:
            recommandations = [
                "Consultation médicale rapide conseillée.",
                "Relecture par un radiologue spécialiste.",
                "Bilan biologique complémentaire (NFS, CRP...).",
                "Surveillance de la saturation en oxygène.",
            ]
        
        st.markdown("### Recommandations suggérées")
        for reco in recommandations:
            st.markdown(f'<div class="reco-item">{reco}</div>', unsafe_allow_html=True)
        
        data_pdf = {
            'nom': nom_patient or "Patient Anonyme",
            'id': f"SC{now.strftime('%Y%m%d%H%M')}",
            'age': age,
            'sexe': sexe,
            'res': prediction,
            'conf': confiance,
            'recos': recommandations
        }
        
        pdf_buffer = generer_pdf(data_pdf)
        
        st.download_button(
            label="📄 Télécharger le Rapport PDF Clinique",
            data=pdf_buffer,
            file_name=f"SenClinique_Rapport_{now.strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("SenClinique AI Pro • Modèle Hybride CNN+ViT (M5) • Développé par Samba Diakho")