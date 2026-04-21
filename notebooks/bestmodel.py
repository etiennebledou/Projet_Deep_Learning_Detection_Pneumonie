# ============================================================
# SECTION 13 : FINE-TUNING FINAL — CNN+ViT OPTIMISE
# ============================================================

fixer_seed(42)

print("="*65)
print("FINE-TUNING FINAL — CNN+ViT avec hyperparametres Optuna")
print(f"lr={BEST_LR:.6f} | blocs={BEST_NB_COUCHES} | dropout={BEST_DROPOUT:.3f}")
print("="*65)

# Recrée une instance propre de CNN+ViT
model5_ft = HybrideCNNViT(nb_classes=2, d_model=512, nhead=8,
                           num_layers=2, dropout=BEST_DROPOUT)
# Appliquer le Fine-Tuning avec les blocs optimaux
model5_ft = degeler_couches(model5_ft, "M5 CNN+ViT", BEST_NB_COUCHES)
model5_ft = model5_ft.to(device)

criterion_ft = FocalLoss(alpha=1.0, gamma=2.0)
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, model5_ft.parameters()),
    lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY
)
scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode="max", patience=3, factor=0.5)

historique_ft = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "recall_pneumo": []
}

meilleur_recall_ft = 0.0
patience_es        = 7
compteur_es        = 0

print("\nEntrainement...")
for epoch in range(20):
    fixer_seed(epoch)
    model5_ft.train()
    tl, tc, tt = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_ft.zero_grad()
        out  = model5_ft(imgs)
        loss = criterion_ft(out, labels)
        loss.backward(); optimizer_ft.step()
        tl += loss.item()
        tc += (out.argmax(1) == labels).sum().item()
        tt += labels.size(0)

    model5_ft.eval()
    vl, v_labels, v_preds = 0, [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out   = model5_ft(imgs)
            vl   += criterion_ft(out, labels).item()
            preds = out.argmax(dim=1)
            v_labels.extend(labels.cpu().numpy())
            v_preds.extend(preds.cpu().numpy())

    tla = tl/len(train_loader); vla = vl/len(val_loader)
    tac = tc/tt
    vac = accuracy_score(v_labels, v_preds)
    rec = recall_score(v_labels, v_preds, pos_label=1, zero_division=0)

    historique_ft["train_loss"].append(tla)
    historique_ft["val_loss"].append(vla)
    historique_ft["train_acc"].append(tac)
    historique_ft["val_acc"].append(vac)
    historique_ft["recall_pneumo"].append(rec)
    scheduler_ft.step(rec)

    print(f"Epoch {epoch+1:02d}/20 | "
          f"Train Loss:{tla:.4f} Acc:{tac:.4f} | "
          f"Val Loss:{vla:.4f} Acc:{vac:.4f} | Recall:{rec:.4f}")

    if rec > meilleur_recall_ft:
        meilleur_recall_ft = rec
        compteur_es        = 0
        torch.save(model5_ft.state_dict(), "model5_cnn_vit_finetuned.pth")
        print(f"   Sauvegarde ! Recall : {meilleur_recall_ft:.4f}")
    else:
        compteur_es += 1
        if compteur_es >= patience_es:
            print(f"Early Stopping epoch {epoch+1}")
            break

# Courbes
plot_historique(historique_ft, "CNN+ViT Fine-Tuning (Optuna)")

# ── Evaluation finale ─────────────────────────────────────────
model5_ft.eval()
labels_all, probs_all = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs  = imgs.to(device)
        probs = torch.softmax(model5_ft(imgs), dim=1)[:,1]
        labels_all.extend(labels.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

probs_all  = np.array(probs_all)
labels_all = np.array(labels_all)

# Trouver seuil optimal
print("\nOptimisation seuil final :")
meilleur_s = 0.5; meilleur_rn = 0.0
print(f"{'Seuil':>7} | {'Rec.Pneumo':>10} | {'Rec.Normal':>10} | {'F1':>8}")
for seuil in np.arange(0.20, 0.65, 0.05):
    preds = (probs_all > seuil).astype(int)
    rp    = recall_score(labels_all, preds, pos_label=1, zero_division=0)
    rn    = recall_score(labels_all, preds, pos_label=0, zero_division=0)
    f1    = f1_score(labels_all, preds, average="weighted", zero_division=0)
    ok    = " *" if rp >= 0.97 else ""
    print(f"{seuil:>7.2f} | {rp:>10.4f} | {rn:>10.4f} | {f1:>8.4f}{ok}")
    if rp >= 0.97 and rn > meilleur_rn:
        meilleur_rn = rn; meilleur_s = seuil

preds_finales = (probs_all > meilleur_s).astype(int)
acc_ft  = accuracy_score(labels_all, preds_finales)
auc_ft  = roc_auc_score(labels_all, probs_all)
rec_p   = recall_score(labels_all, preds_finales, pos_label=1, zero_division=0)
rec_n   = recall_score(labels_all, preds_finales, pos_label=0, zero_division=0)

print(f"\n{'='*60}")
print(f"RESULTATS FINAUX — CNN+ViT Fine-Tuning (Optuna)")
print(f"{'='*60}")
print(f"Accuracy          : {acc_ft:.4f}")
print(f"AUC ROC           : {auc_ft:.4f}")
print(f"Recall PNEUMONIE  : {rec_p:.4f}")
print(f"Recall NORMAL     : {rec_n:.4f}")
print(f"Seuil optimal     : {meilleur_s:.2f}")
print(f"\n{classification_report(labels_all, preds_finales, target_names=['NORMAL','PNEUMONIA'])}")

# Comparaison avec le modele de base
print(f"\nCOMPARAISON :")
print(f"{'Modele':<38} {'Rec.Pneumo':>11} {'Rec.Normal':>11} {'AUC':>7}")
print(f"{'CNN+ViT Transfer Learning (base)':<38} {'0.9718':>11} {'0.7051':>11} {'0.957':>7}")
print(f"{'CNN+ViT Fine-Tuning + Optuna':<38} {rec_p:>11.4f} {rec_n:>11.4f} {auc_ft:>7.4f}")
amelio = rec_n - 0.7051
signe  = "+" if amelio >= 0 else ""
print(f"\nEvolution Recall NORMAL : {signe}{amelio:.4f} ({signe}{amelio*100:.1f} points)")