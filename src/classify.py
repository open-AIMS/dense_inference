import joblib


def import_model(model_file_name):
    svc_std, sc, le = joblib.load(model_file_name)
    return svc_std, sc, le

def classify(vectors, model_pth):
    model, scaler, label_encoder = import_model(model_pth)

    scaled = scaler.transform(vectors)
    pred_code = model.predict(scaled)

    pred_labels = label_encoder.inverse_transform(pred_code)
    return pred_code, pred_labels