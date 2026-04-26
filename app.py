"""
Brine Shrimp Detector and Counter - Flask Web App
Allows users to upload 1-10 petri dish images and receive:
- segmented output images
- per-image brine shrimp counts
- total shrimp count
- downloadable CSV summary
"""

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    send_from_directory,
)

from web_inference import (
    get_device,
    load_trained_model,
    process_uploaded_files,
    WEB_RESULTS_DIR,
    UPLOADS_DIR,
)

""" FLASK APP SETUP """
app = Flask(__name__)
app.secret_key = "brine_shrimp_final_project_secret_key"

# Load model once at startup
DEVICE = get_device()
MODEL, DEVICE = load_trained_model(DEVICE)

# In-memory storage for recent batch results
# Fine for a class project/local demo
BATCH_RESULTS = {}


""" ROUTES """
@app.route("/", methods=["GET"])
def index():
    """Render the homepage upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept 1-10 uploaded images, run inference, and render the results page.
    """
    files = request.files.getlist("images")

    try:
        batch_data = process_uploaded_files(
            files=files,
            model=MODEL,
            device=DEVICE,
        )

        batch_id = batch_data["batch_id"]
        BATCH_RESULTS[batch_id] = batch_data

        return render_template(
            "results.html",
            batch_id=batch_data["batch_id"],
            results=batch_data["results"],
            total_count=batch_data["total_count"],
            num_uploaded=batch_data["num_uploaded"],
            csv_name=batch_data["csv_name"],
            confidence_threshold=batch_data["confidence_threshold"],
        )

    except Exception as e:
        flash(str(e), "error")
        return redirect(url_for("index"))


@app.route("/results/<batch_id>", methods=["GET"])
def show_results(batch_id):
    """
    Re-open a recently processed results page from in-memory storage.
    """
    batch_data = BATCH_RESULTS.get(batch_id)

    if not batch_data:
        flash("That result session is no longer available. Please upload your images again.", "error")
        return redirect(url_for("index"))

    return render_template(
        "results.html",
        batch_id=batch_data["batch_id"],
        results=batch_data["results"],
        total_count=batch_data["total_count"],
        num_uploaded=batch_data["num_uploaded"],
        csv_name=batch_data["csv_name"],
        confidence_threshold=batch_data["confidence_threshold"],
    )


@app.route("/uploads/<filename>", methods=["GET"])
def uploaded_file(filename):
    """
    Serve uploaded images so they can be displayed on the results page.
    """
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/download/<csv_name>", methods=["GET"])
def download_csv(csv_name):
    """
    Download the generated CSV summary for a processed batch.
    """
    csv_path = WEB_RESULTS_DIR / csv_name

    if not csv_path.exists():
        flash("CSV file not found.", "error")
        return redirect(url_for("index"))

    return send_file(
        csv_path,
        as_attachment=True,
        download_name=csv_path.name,
        mimetype="text/csv",
    )


@app.route("/health", methods=["GET"])
def health():
    """
    Simple health check route.
    """
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": True,
    }


""" RUN APP """
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)