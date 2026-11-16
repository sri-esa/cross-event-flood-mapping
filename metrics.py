import numpy as np

def compute_shared_metrics(pred_mask, true_mask, pixel_area_m2=100.0):
    pred_arr = np.asarray(pred_mask)
    true_arr = np.asarray(true_mask)

    # Strictly exclude ignore (-1), 255 nodata, and NaNs from intersection and union
    valid = (true_arr != -1) & (true_arr != 255) & np.isfinite(true_arr) & np.isfinite(pred_arr)

    p = pred_arr[valid].astype(bool)
    t = (true_arr[valid] == 1)

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, np.logical_not(t)).sum())
    tn = int(np.logical_and(np.logical_not(p), np.logical_not(t)).sum())
    fn = int(np.logical_and(np.logical_not(p), t).sum())

    union = tp + fp + fn
    gt_flood = tp + fn
    pred_flood = tp + fp
    total_valid = tp + fp + tn + fn

    iou = tp / union if union > 0 else (1.0 if gt_flood == 0 and pred_flood == 0 else 0.0)
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if gt_flood == 0 and pred_flood == 0 else 0.0)
    precision = tp / pred_flood if pred_flood > 0 else (1.0 if gt_flood == 0 else 0.0)
    recall = tp / gt_flood if gt_flood > 0 else (1.0 if pred_flood == 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / total_valid if total_valid > 0 else np.nan

    gt_area_km2 = (gt_flood * pixel_area_m2) / 1e6
    pred_area_km2 = (pred_flood * pixel_area_m2) / 1e6

    return {
        "IoU": round(float(iou), 5),
        "Precision": round(float(precision), 5),
        "Recall": round(float(recall), 5),
        "F1": round(float(f1), 5),
        "GT_Area_km2": round(float(gt_area_km2), 4),
        "Pred_Area_km2": round(float(pred_area_km2), 4),
        "Specificity": round(float(specificity), 5) if np.isfinite(specificity) else np.nan,
        "Accuracy": round(float(accuracy), 5) if np.isfinite(accuracy) else np.nan,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }
