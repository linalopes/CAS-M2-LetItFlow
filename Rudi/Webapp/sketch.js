let mic, meyda, model;
let audioData = [];
let recordingDuration = 3;
let sampleRate = 16000;
let mfccFeatures = [];
let mfccBufferSize = 400; // Number of time steps expected by the model
let mfccFeatureSize = 13; // Number of features per time step expected by the model
let micButton, micOffButton, micSensitivitySlider;
let waterSpeed = 0;
let waterSpeedHistory = [];
let modelReady = false;
let amplitudeSpectrum = new Array(64).fill(0); // Default array for visualization

async function loadTFModel() {
    try {
        console.log("Attempting to load model from: model.json");
        model = await tf.loadGraphModel('model.json');
        console.log('Model Loaded Successfully');
        modelReady = true;
        micButton.removeAttribute('disabled');
    } catch (error) {
        console.error("Error loading model:", error);
        alert("Failed to load the model. Check the console for details.");
    }
}

function extractAudioFeatures(features) {
    if (features) {
        if (features.mfcc) {
            // Flatten the MFCC feature vector to ensure we have the correct structure
            mfccFeatures.push(features.mfcc);
            console.log("Extracted MFCC:", features.mfcc);

            // Ensure that we accumulate exactly 400 timesteps, each with 13 features
            if (mfccFeatures.length >= mfccBufferSize) {
                // Slice to make sure it has exactly 400 time steps
                let reshapedFeatures = mfccFeatures.slice(0, mfccBufferSize);

                makePrediction(reshapedFeatures);
                mfccFeatures = []; // Reset after making a prediction
            }
        }
        if (features.amplitudeSpectrum) {
            amplitudeSpectrum = features.amplitudeSpectrum;
        } else {
            console.log("No amplitude spectrum extracted.");
        }
    }
}

async function makePrediction(reshapedFeatures) {
    if (model) {
        // Check if reshapedFeatures have the correct length and reshape them to [1, 400, 13]
        console.log("Extracted MFCC Features:", reshapedFeatures);
        console.log("Feature Length:", reshapedFeatures.length);

        if (Array.isArray(reshapedFeatures) && reshapedFeatures.length === mfccBufferSize) {
            // Create a tensor of shape [1, 400, 13] to match the model input
            let inputTensor = tf.tensor3d([reshapedFeatures], [1, mfccBufferSize, mfccFeatureSize]);
            const prediction = await model.predict(inputTensor);
            let result = prediction.dataSync();

            waterSpeed = result[0];
            waterSpeedHistory.push(waterSpeed);

            if (waterSpeedHistory.length > 100) {
                waterSpeedHistory.shift();
            }

            tf.dispose([inputTensor, prediction]);
        } else {
            console.error("Input features do not match the expected shape. Expected 400 time steps, each with 13 features.");
        }
    }
}

function setup() {
    createCanvas(windowWidth, windowHeight);
    background(200);
    textSize(32);
    fill(0);

    micButton = createButton('Activate Microphone');
    micButton.position(20, 20);
    micButton.style('font-size', '24px');
    micButton.mousePressed(activateMic);
    micButton.attribute('disabled', '');

    micOffButton = createButton('Turn Off Microphone');
    micOffButton.position(220, 20);
    micOffButton.style('font-size', '24px');
    micOffButton.mousePressed(turnOffMic);
    micOffButton.attribute('disabled', '');

    micSensitivitySlider = createSlider(0, 10, 1, 0.1);
    micSensitivitySlider.position(20, 70);
    micSensitivitySlider.style('width', '200px');

    loadTFModel();
}

function activateMic() {
    if (modelReady && (!mic || !mic.enabled)) {
        mic = new p5.AudioIn();
        mic.start(() => {
            console.log("Microphone activated");

            try {
                // Use audio context directly to get the mic stream
                let audioContext = getAudioContext();
                let micSourceNode = audioContext.createMediaStreamSource(mic.stream);

                meyda = Meyda.createMeydaAnalyzer({
                    audioContext: audioContext,
                    source: micSourceNode, // Pass the source node directly
                    bufferSize: 2048,
                    featureExtractors: ['mfcc', 'amplitudeSpectrum'],
                    callback: extractAudioFeatures
                });

                meyda.start();
                console.log("Meyda analyzer started"); // Log when Meyda starts
            } catch (error) {
                console.error("Error starting Meyda:", error);
            }
        });

        micButton.attribute('disabled', '');
        micOffButton.removeAttribute('disabled');
    } else if (!modelReady) {
        console.log("Model is still loading. Please wait...");
    }
}

function turnOffMic() {
    if (mic && mic.enabled) {
        mic.stop();
        mic = null;
        console.log("Microphone turned off");
    }

    if (meyda) {
        try {
            meyda.stop();
            console.log("Meyda analyzer stopped");
        } catch (error) {
            console.log("Error stopping Meyda:", error);
        }
    }

    waterSpeed = 0;
    micButton.removeAttribute('disabled');
    micOffButton.attribute('disabled', '');
}

function draw() {
    background(200);

    if (mic) {
        mic.amp(micSensitivitySlider.value() * 10); // Boost mic input by slider value
        let micLevel = mic.getLevel();

        textSize(24);
        textAlign(LEFT, BOTTOM);
        text(`Mic Level: ${nf(micLevel, 0, 6)}`, 20, height - 20);
    }

    if (!modelReady) {
        fill(255, 0, 0);
        textSize(30);
        text("Loading model... Please wait.", width / 2 - 150, height / 2);
        return;
    }

    textSize(40);
    fill(0);
    text(`Water Speed: ${waterSpeed.toFixed(2)} units`, 20, height / 2);

    drawWaterSpeedGraph();
    drawAmplitudeSpectrum(); // Visualize the amplitude spectrum
}

function drawWaterSpeedGraph() {
    stroke(0);
    noFill();
    beginShape();
    for (let i = 0; i < waterSpeedHistory.length; i++) {
        let x = map(i, 0, waterSpeedHistory.length, 0, width);
        let y = map(waterSpeedHistory[i], 0, 10, height - 100, height - 300);
        vertex(x, y);
    }
    endShape();
}

function drawAmplitudeSpectrum() {
    if (amplitudeSpectrum.length > 0) {
        stroke(0);
        fill(0, 100, 255, 150);
        let bandWidth = width / amplitudeSpectrum.length;
        for (let i = 0; i < amplitudeSpectrum.length; i++) {
            let value = Math.max(amplitudeSpectrum[i], 0.01); // Set a minimum threshold
            let x = i * bandWidth;
            let h = map(value, 0, 1, 0, height / 2);
            rect(x, height - h - 150, bandWidth, h);
        }
    }
}
