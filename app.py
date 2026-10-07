import os
import time
import sqlite3
import threading
from datetime import datetime

import cv2
import numpy as np
import pytz

from flask import (
    Flask,
    render_template_string,
    jsonify,
    send_from_directory,
    request
)

from ultralytics import YOLO


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Maximum upload size per frame request
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAVE_FOLDER = os.path.join(
    BASE_DIR,
    "detected_frames"
)

DB_NAME = os.path.join(
    BASE_DIR,
    "detections.db"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

os.makedirs(SAVE_FOLDER, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def init_db():

    with sqlite3.connect(DB_NAME) as conn:

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                person_count INTEGER,
                detection_time_ist TEXT
            )
        """)

        conn.commit()


def log_detection(image_path, count):

    ist = pytz.timezone("Asia/Kolkata")

    current_time_ist = datetime.now(
        ist
    ).strftime(
        "%Y-%m-%d %I:%M:%S %p"
    )

    with sqlite3.connect(DB_NAME) as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO person_logs
            (
                image_path,
                person_count,
                detection_time_ist
            )
            VALUES (?, ?, ?)
            """,
            (
                image_path,
                int(count),
                current_time_ist
            )
        )

        conn.commit()


init_db()


# ============================================================
# LOAD YOLO MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )


print("Loading YOLOv8 model...")

model = YOLO(MODEL_PATH)

print("YOLOv8 model loaded successfully.")


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE = 0.50

LOG_COOLDOWN = 10

MAX_HISTORY_IMAGES = 300

last_log_time = 0

model_lock = threading.Lock()


# ============================================================
# CLEAN OLD DETECTION IMAGES
# ============================================================

def cleanup_old_images():

    try:

        files = []

        for filename in os.listdir(SAVE_FOLDER):

            filepath = os.path.join(
                SAVE_FOLDER,
                filename
            )

            if os.path.isfile(filepath):

                files.append(
                    (
                        os.path.getmtime(filepath),
                        filepath
                    )
                )

        files.sort(
            key=lambda x: x[0],
            reverse=True
        )

        old_files = files[MAX_HISTORY_IMAGES:]

        for _, filepath in old_files:

            try:
                os.remove(filepath)

            except Exception as error:

                print(
                    "Unable to remove old image:",
                    error
                )

    except Exception as error:

        print(
            "Cleanup error:",
            error
        )


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_persons(frame):

    global last_log_time

    person_count = 0

    annotated_frame = frame.copy()

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    with model_lock:

        results = model.predict(
            frame,
            classes=[0],
            conf=CONFIDENCE,
            verbose=False,
            device="cpu"
        )

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    for result in results:

        boxes = result.boxes

        for box in boxes:

            person_count += 1

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordinates

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (99, 102, 241),
                2
            )

            # ------------------------------------------------
            # PERSON LABEL
            # ------------------------------------------------

            label = (
                f"Person "
                f"{confidence * 100:.1f}%"
            )

            cv2.rectangle(
                annotated_frame,
                (
                    x1,
                    max(0, y1 - 30)
                ),
                (
                    x1 + 145,
                    y1
                ),
                (99, 102, 241),
                -1
            )

            cv2.putText(
                annotated_frame,
                label,
                (
                    x1 + 5,
                    max(20, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    # ========================================================
    # PERSON COUNT
    # ========================================================

    count_text = (
        f"Persons: {person_count}"
    )

    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (230, 65),
        (15, 23, 42),
        -1
    )

    cv2.putText(
        annotated_frame,
        count_text,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ========================================================
    # SAVE DETECTION
    # ========================================================

    current_time = time.time()

    if (
        person_count > 0
        and
        (current_time - last_log_time)
        > LOG_COOLDOWN
    ):

        ist = pytz.timezone(
            "Asia/Kolkata"
        )

        timestamp_str = datetime.now(
            ist
        ).strftime(
            "%Y%m%d_%I%M%S_%p"
        )

        img_filename = (
            f"person_{timestamp_str}.jpg"
        )

        img_filepath = os.path.join(
            SAVE_FOLDER,
            img_filename
        )

        # Save annotated image

        cv2.imwrite(
            img_filepath,
            annotated_frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                85
            ]
        )

        # Save database record

        relative_path = os.path.join(
            "detected_frames",
            img_filename
        )

        log_detection(
            relative_path,
            person_count
        )

        last_log_time = current_time

        cleanup_old_images()

    return annotated_frame, person_count


# ============================================================
# HOME PAGE
# ============================================================

HTML_TEMPLATE = """

<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        VisionAI Fast - Person Detection
    </title>

    <!-- Tailwind -->

    <script src="https://cdn.tailwindcss.com"></script>

    <!-- Font Awesome -->

    <link
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
        rel="stylesheet"
    >

</head>


<body
    class="bg-gray-50 min-h-screen flex flex-col font-sans"
>


<!-- ======================================================
     NAVBAR
======================================================= -->

<nav
    class="bg-slate-900 text-white shadow-xl"
>

    <div
        class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
    >

        <div
            class="flex items-center justify-between h-16"
        >

            <!-- LOGO -->

            <div
                class="flex items-center gap-3"
            >

                <i
                    class="fa-solid fa-bolt text-yellow-400 text-2xl"
                ></i>

                <span
                    class="font-bold text-xl tracking-wider"
                >

                    VisionAI

                    <span
                        class="text-yellow-400"
                    >
                        Fast
                    </span>

                </span>

            </div>


            <!-- STATUS -->

            <div
                id="status"
                class="text-sm text-gray-300 font-medium"
            >

                <i
                    class="fa-solid fa-circle text-gray-500 mr-2"
                ></i>

                Camera Off

            </div>

        </div>

    </div>

</nav>


<!-- ======================================================
     MAIN
======================================================= -->

<main
    class="flex-grow w-full max-w-7xl mx-auto px-3 sm:px-6 py-6"
>


<!-- ======================================================
     CAMERA SECTION
======================================================= -->

<section
    class="bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-100"
>


    <!-- HEADER -->

    <div
        class="bg-gradient-to-r from-slate-800 to-slate-700 p-4 text-center border-b-4 border-yellow-400"
    >

        <h2
            class="text-xl font-bold text-white"
        >

            <i
                class="fa-solid fa-video mr-2 text-yellow-400"
            ></i>

            Live Person Detection

        </h2>

        <p
            class="text-gray-300 text-sm mt-1"
        >

            Browser Camera + YOLOv8 Person Detection

        </p>

    </div>


    <!-- CAMERA -->

    <div
        class="p-3 sm:p-6 bg-slate-950"
    >

        <div
            class="relative max-w-4xl mx-auto"
        >

            <!-- CAMERA VIDEO -->

            <video
                id="camera"
                autoplay
                playsinline
                muted
                class="w-full rounded-lg shadow-lg border-2 border-slate-700 bg-black"
            ></video>


            <!-- DETECTION CANVAS -->

            <canvas
                id="canvas"
                class="hidden"
            ></canvas>


            <!-- RESULT IMAGE -->

            <img
                id="resultImage"
                class="absolute top-0 left-0 w-full h-full object-contain rounded-lg pointer-events-none"
                style="display:none;"
                alt="YOLO Detection"
            >


            <!-- PERSON COUNT -->

            <div
                class="absolute top-4 left-4 bg-slate-900/90 text-white px-4 py-2 rounded-xl shadow-lg"
            >

                <div
                    class="text-xs text-gray-400"
                >
                    DETECTED PERSONS
                </div>

                <div
                    id="personCount"
                    class="text-3xl font-bold text-green-400"
                >
                    0
                </div>

            </div>

        </div>


        <!-- CONTROLS -->

        <div
            class="flex flex-wrap justify-center gap-3 mt-5"
        >

            <button
                id="startButton"
                onclick="startCamera()"
                class="bg-green-500 hover:bg-green-400 text-white px-6 py-3 rounded-xl font-semibold transition shadow-lg"
            >

                <i
                    class="fa-solid fa-camera mr-2"
                ></i>

                Start Camera

            </button>


            <button
                id="stopButton"
                onclick="stopCamera()"
                class="bg-red-500 hover:bg-red-400 text-white px-6 py-3 rounded-xl font-semibold transition shadow-lg"
            >

                <i
                    class="fa-solid fa-stop mr-2"
                ></i>

                Stop Camera

            </button>

        </div>


        <!-- INFO -->

        <div
            class="text-center mt-4 text-gray-400 text-sm"
        >

            Your browser camera is processed by the YOLOv8 model on the server.

        </div>

    </div>

</section>


<!-- ======================================================
     SUMMARY
======================================================= -->

<section
    class="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6"
>


    <!-- TOTAL -->

    <div
        class="bg-white rounded-xl shadow-md border border-gray-100 p-5"
    >

        <div
            class="flex items-center justify-between"
        >

            <div>

                <p
                    class="text-gray-500 text-sm"
                >
                    Total Detections
                </p>

                <h3
                    id="totalDetections"
                    class="text-3xl font-bold text-slate-800 mt-1"
                >
                    0
                </h3>

            </div>

            <div
                class="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center"
            >

                <i
                    class="fa-solid fa-list text-indigo-600 text-xl"
                ></i>

            </div>

        </div>

    </div>


    <!-- LATEST COUNT -->

    <div
        class="bg-white rounded-xl shadow-md border border-gray-100 p-5"
    >

        <div
            class="flex items-center justify-between"
        >

            <div>

                <p
                    class="text-gray-500 text-sm"
                >
                    Latest Person Count
                </p>

                <h3
                    id="latestCount"
                    class="text-3xl font-bold text-green-600 mt-1"
                >
                    0
                </h3>

            </div>

            <div
                class="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center"
            >

                <i
                    class="fa-solid fa-user text-green-600 text-xl"
                ></i>

            </div>

        </div>

    </div>


    <!-- LATEST TIME -->

    <div
        class="bg-white rounded-xl shadow-md border border-gray-100 p-5"
    >

        <div
            class="flex items-center justify-between"
        >

            <div>

                <p
                    class="text-gray-500 text-sm"
                >
                    Latest Detection
                </p>

                <h3
                    id="latestTime"
                    class="text-sm sm:text-base font-bold text-slate-800 mt-2"
                >
                    No detections
                </h3>

            </div>

            <div
                class="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center"
            >

                <i
                    class="fa-solid fa-clock text-yellow-600 text-xl"
                ></i>

            </div>

        </div>

    </div>

</section>


<!-- ======================================================
     DETECTION HISTORY
======================================================= -->

<section
    class="bg-white rounded-2xl shadow-2xl border border-gray-100 overflow-hidden mt-6"
>


    <!-- HEADER -->

    <div
        class="bg-gradient-to-r from-slate-800 to-slate-700 px-5 py-4 flex items-center justify-between"
    >

        <div>

            <h2
                class="text-lg font-bold text-white"
            >

                <i
                    class="fa-solid fa-table mr-2 text-yellow-400"
                ></i>

                Detection History

            </h2>

            <p
                class="text-gray-300 text-sm mt-1"
            >
                Latest person detections
            </p>

        </div>


        <button
            onclick="loadDetections()"
            class="bg-yellow-400 hover:bg-yellow-300 text-slate-900 px-4 py-2 rounded-lg font-semibold text-sm transition"
        >

            <i
                class="fa-solid fa-rotate mr-1"
            ></i>

            Refresh

        </button>

    </div>


    <!-- TABLE -->

    <div
        class="overflow-x-auto"
    >

        <table
            class="w-full text-left"
        >

            <thead
                class="bg-slate-100 border-b border-slate-200"
            >

                <tr>

                    <th
                        class="px-5 py-4 text-sm font-bold text-slate-700"
                    >
                        ID
                    </th>

                    <th
                        class="px-5 py-4 text-sm font-bold text-slate-700"
                    >
                        Detection Image
                    </th>

                    <th
                        class="px-5 py-4 text-sm font-bold text-slate-700"
                    >
                        Persons
                    </th>

                    <th
                        class="px-5 py-4 text-sm font-bold text-slate-700"
                    >
                        Detection Time (IST)
                    </th>

                </tr>

            </thead>


            <tbody
                id="detectionTable"
                class="divide-y divide-gray-100"
            >

                <tr>

                    <td
                        colspan="4"
                        class="px-5 py-8 text-center text-gray-500"
                    >

                        Loading detections...

                    </td>

                </tr>

            </tbody>

        </table>

    </div>

</section>


</main>


<!-- ======================================================
     FOOTER
======================================================= -->

<footer
    class="bg-slate-900 text-slate-400 py-6 text-center text-sm border-t border-slate-800"
>

    <p>
        &copy; 2026 VisionAI Fast. All rights reserved.
    </p>

</footer>


<!-- ======================================================
     JAVASCRIPT
======================================================= -->

<script>


// ========================================================
// VARIABLES
// ========================================================

let stream = null;

let detectionInterval = null;

let processing = false;

const video =
    document.getElementById("camera");

const canvas =
    document.getElementById("canvas");

const resultImage =
    document.getElementById("resultImage");

const personCount =
    document.getElementById("personCount");

const statusElement =
    document.getElementById("status");


// ========================================================
// START CAMERA
// ========================================================

async function startCamera() {

    try {

        stream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        },

                        facingMode: "user"
                    },

                    audio: false
                }
            );


        video.srcObject = stream;


        statusElement.innerHTML = `
            <i
                class="fa-solid fa-circle text-green-400 mr-2 animate-pulse"
            ></i>
            Camera Active
        `;


        startDetection();


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        alert(
            "Unable to access camera. Please allow camera permission in your browser."
        );

    }

}


// ========================================================
// STOP CAMERA
// ========================================================

function stopCamera() {

    if (detectionInterval) {

        clearInterval(
            detectionInterval
        );

        detectionInterval = null;

    }


    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        stream = null;

    }


    video.srcObject = null;


    resultImage.style.display = "none";


    personCount.textContent = "0";


    statusElement.innerHTML = `
        <i
            class="fa-solid fa-circle text-gray-500 mr-2"
        ></i>
        Camera Off
    `;

}


// ========================================================
// START DETECTION
// ========================================================

function startDetection() {

    if (detectionInterval) {

        clearInterval(
            detectionInterval
        );

    }


    // Send one frame every 700 ms

    detectionInterval =
        setInterval(
            sendFrame,
            700
        );

}


// ========================================================
// SEND FRAME TO SERVER
// ========================================================

async function sendFrame() {

    if (!stream) {
        return;
    }


    if (processing) {
        return;
    }


    if (
        video.readyState
        <
        HTMLMediaElement.HAVE_ENOUGH_DATA
    ) {
        return;
    }


    processing = true;


    try {

        const width = 640;

        const height = 480;


        canvas.width = width;

        canvas.height = height;


        const context =
            canvas.getContext("2d");


        context.drawImage(
            video,
            0,
            0,
            width,
            height
        );


        const blob =
            await new Promise(
                resolve =>
                    canvas.toBlob(
                        resolve,
                        "image/jpeg",
                        0.75
                    )
            );


        if (!blob) {

            processing = false;

            return;

        }


        const formData =
            new FormData();


        formData.append(
            "frame",
            blob,
            "frame.jpg"
        );


        const response =
            await fetch(
                "/detect_frame",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                "Detection request failed"
            );

        }


        const data =
            await response.json();


        // Update count

        personCount.textContent =
            data.person_count;


        // Display processed frame

        if (data.image) {

            resultImage.src =
                "data:image/jpeg;base64,"
                +
                data.image;

            resultImage.style.display =
                "block";

        }


        // Update status

        statusElement.innerHTML = `
            <i
                class="fa-solid fa-circle text-green-400 mr-2 animate-pulse"
            ></i>
            AI Detection Active
        `;


    } catch (error) {

        console.error(
            "Frame processing error:",
            error
        );

    }


    processing = false;

}


// ========================================================
// LOAD DETECTIONS
// ========================================================

async function loadDetections() {

    const table =
        document.getElementById(
            "detectionTable"
        );


    try {

        const response =
            await fetch(
                "/api/detections"
            );


        if (!response.ok) {

            throw new Error(
                "Failed to load detections"
            );

        }


        const detections =
            await response.json();


        // ==================================================
        // EMPTY STATE
        // ==================================================

        if (
            detections.length === 0
        ) {

            table.innerHTML = `

                <tr>

                    <td
                        colspan="4"
                        class="px-5 py-10 text-center text-gray-500"
                    >

                        <div
                            class="flex flex-col items-center gap-2"
                        >

                            <i
                                class="fa-solid fa-inbox text-3xl text-gray-300"
                            ></i>

                            <span>
                                No detections available
                            </span>

                        </div>

                    </td>

                </tr>

            `;


            document
                .getElementById(
                    "totalDetections"
                )
                .textContent = "0";


            document
                .getElementById(
                    "latestCount"
                )
                .textContent = "0";


            document
                .getElementById(
                    "latestTime"
                )
                .textContent =
                "No detections";


            return;

        }


        // ==================================================
        // TABLE
        // ==================================================

        table.innerHTML = "";


        detections.forEach(
            detection => {

                const row =
                    document.createElement(
                        "tr"
                    );


                row.className =
                    "hover:bg-gray-50 transition";


                row.innerHTML = `

                    <td
                        class="px-5 py-4"
                    >

                        <span
                            class="font-semibold text-slate-700"
                        >
                            #${detection.id}
                        </span>

                    </td>


                    <td
                        class="px-5 py-4"
                    >

                        <img
                            src="${detection.image_url}"
                            alt="Detection"
                            class="w-24 h-16 object-cover rounded-lg border border-gray-200 shadow-sm"
                            onerror="this.style.display='none'"
                        >

                    </td>


                    <td
                        class="px-5 py-4"
                    >

                        <span
                            class="inline-flex items-center gap-2 bg-indigo-100 text-indigo-700 px-3 py-1 rounded-full font-semibold text-sm"
                        >

                            <i
                                class="fa-solid fa-user"
                            ></i>

                            ${detection.person_count}

                        </span>

                    </td>


                    <td
                        class="px-5 py-4 text-sm text-gray-600"
                    >

                        ${detection.detection_time}

                    </td>

                `;


                table.appendChild(
                    row
                );

            }
        );


        // ==================================================
        // SUMMARY
        // ==================================================

        document
            .getElementById(
                "totalDetections"
            )
            .textContent =
            detections.length;


        document
            .getElementById(
                "latestCount"
            )
            .textContent =
            detections[0].person_count;


        document
            .getElementById(
                "latestTime"
            )
            .textContent =
            detections[0].detection_time;


    } catch (error) {

        console.error(
            "Detection loading error:",
            error
        );


        table.innerHTML = `

            <tr>

                <td
                    colspan="4"
                    class="px-5 py-8 text-center text-red-500"
                >

                    <i
                        class="fa-solid fa-triangle-exclamation mr-2"
                    ></i>

                    Unable to load detection history

                </td>

            </tr>

        `;

    }

}


// ========================================================
// INITIAL LOAD
// ========================================================

loadDetections();


// ========================================================
// AUTO REFRESH HISTORY
// ========================================================

setInterval(
    loadDetections,
    5000
);


// ========================================================
// STOP CAMERA WHEN PAGE IS CLOSED
// ========================================================

window.addEventListener(
    "beforeunload",
    stopCamera
);


</script>


</body>

</html>

"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML_TEMPLATE
    )


# ============================================================
# DETECT FRAME API
# ============================================================

@app.route(
    "/detect_frame",
    methods=["POST"]
)
def detect_frame():

    try:

        if "frame" not in request.files:

            return jsonify(
                {
                    "error":
                    "No frame received"
                }
            ), 400


        file = request.files["frame"]


        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = file.read()


        if not image_bytes:

            return jsonify(
                {
                    "error":
                    "Empty frame"
                }
            ), 400


        # Convert bytes to NumPy

        image_array = np.frombuffer(
                image_bytes,
                np.uint8
            )


        frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


        if frame is None:

            return jsonify(
                {
                    "error":
                    "Invalid image"
                }
            ), 400


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        frame = cv2.resize(
                frame,
                (640, 480)
            )


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        annotated_frame, person_count = \
            detect_persons(
                frame
            )


        # ----------------------------------------------------
        # ENCODE RESULT
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
                ".jpg",
                annotated_frame,
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    75
                ]
            )


        if not success:

            return jsonify(
                {
                    "error":
                    "Unable to encode result"
                }
            ), 500


        # ----------------------------------------------------
        # BASE64
        # ----------------------------------------------------

        import base64

        encoded_image = base64.b64encode(
                buffer.tobytes()
            ).decode(
                "utf-8"
            )


        return jsonify(
            {
                "success": True,
                "person_count":
                    person_count,
                "image":
                    encoded_image
            }
        )


    except Exception as error:

        print(
            "Detection error:",
            error
        )


        return jsonify(
            {
                "success": False,
                "error":
                    str(error)
            }
        ), 500


# ============================================================
# DETECTION HISTORY API
# ============================================================

@app.route(
    "/api/detections"
)
def get_detections():

    try:

        with sqlite3.connect(
            DB_NAME
        ) as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            cursor = conn.cursor()


            cursor.execute(
                """
                SELECT
                    id,
                    image_path,
                    person_count,
                    detection_time_ist
                FROM person_logs
                ORDER BY id DESC
                LIMIT 20
                """
            )


            rows = cursor.fetchall()


        detections = []


        for row in rows:

            image_path = row["image_path"]


            filename = os.path.basename(
                    image_path
                )


            detections.append(
                {
                    "id":
                        row["id"],

                    "person_count":
                        row["person_count"],

                    "detection_time":
                        row[
                            "detection_time_ist"
                        ],

                    "image_url":
                        f"/detected_frames/{filename}"
                }
            )


        return jsonify(
            detections
        )


    except Exception as error:

        print(
            "Database error:",
            error
        )


        return jsonify(
            {
                "error":
                    "Unable to load detections"
            }
        ), 500


# ============================================================
# SERVE DETECTED IMAGES
# ============================================================

@app.route(
    "/detected_frames/<path:filename>"
)
def detected_frame(filename):

    return send_from_directory(
        SAVE_FOLDER,
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health"
)
def health():

    return jsonify(
        {
            "status":
                "healthy",

            "service":
                "VisionAI Fast",

            "model":
                "YOLOv8n",

            "detection":
                "person",

            "server_time":
                datetime.now().isoformat()
        }
    )


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(
    413
)
def request_too_large(error):

    return jsonify(
        {
            "error":
                "Frame is too large"
        }
    ), 413


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )