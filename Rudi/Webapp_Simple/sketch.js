let mic, meyda, model;
let mfccFeatures = [];
let micButton, micOffButton, micSensitivitySlider;
let modelReady = false;
let waterFlowDetected = false;
let waterFlowHistory = [];
let micLevel = 0; // For displaying microphone input level

async function loadTFModel() {
    try {
        console.log("Attempting to load model from: model.json");
        model = await tf.loadGraphModel('model.json');
        console.log('Model Loaded Successfully');

        // Log the input and output tensors directly
        console.log("Model Inputs:", model.inputs);
        console.log("Model Outputs:", model.outputs);

        modelReady = true;
        micButton.removeAttribute('disabled');
    } catch (error) {
        console.error("Error loading model:", error);
        alert("Failed to load the model. Check the console for details.");
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
                let audioContext = getAudioContext();
                let micSourceNode = audioContext.createMediaStreamSource(mic.stream);

                meyda = Meyda.createMeydaAnalyzer({
                    audioContext: audioContext,
                    source: micSourceNode,
                    bufferSize: 2048,
                    featureExtractors: ['mfcc'],
                    callback: extractAudioFeatures
                });

                meyda.start();
                console.log("Meyda analyzer started");
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

    micButton.removeAttribute('disabled');
    micOffButton.attribute('disabled', '');
}

function extractAudioFeatures(features) {
    if (features && features.mfcc) {
        mfccFeatures.push(features.mfcc);
        console.log("Extracted MFCC:", features.mfcc);

        if (mfccFeatures.length >= 100) { // Assuming 100 timesteps
            makePrediction(mfccFeatures.slice(0, 100));
            mfccFeatures = [];
        }
    }
}

async function makePrediction(reshapedFeatures) {
    if (model) {
        // Create input tensor with the correct shape
        let inputTensor = tf.tensor3d([reshapedFeatures], [1, 100, 13]);

        // Debug log: Print the shape of the input tensor
        console.log("Input Tensor Shape:", inputTensor.shape);
        console.log("Input Tensor Values (First 5 Timesteps):", inputTensor.arraySync()[0].slice(0, 5));

        try {
            // Execute the model using executeAsync for dynamic ops
            const prediction = await model.executeAsync(
                { 'keras_tensor_15': inputTensor }, // Input tensor with the correct name
                ['Identity:0'] // Use Identity:0 to specify the output node correctly
            );

            // Get the output data
            let result = prediction.dataSync();

            // Debug log: Print the prediction result
            console.log("Prediction Result:", result);

            // Check if the result indicates water flow detection
            waterFlowDetected = result[0] > 0.5; // Adjust threshold as necessary
            waterFlowHistory.push(waterFlowDetected);

            if (waterFlowHistory.length > 100) {
                waterFlowHistory.shift(); // Maintain a history buffer
            }

            // Dispose tensors to free memory
            tf.dispose([inputTensor, prediction]);
        } catch (error) {
            console.error("Error during prediction:", error);
        }
    }
}








function draw() {
    background(200);
    textSize(30);

    if (!modelReady) {
        fill(255, 0, 0);
        text("Loading model... Please wait.", width / 2 - 150, height / 2);
        return;
    }

    fill(0);
    text("Model is ready. Use the microphone buttons.", 20, 100);

    // Display microphone level
    if (mic) {
        mic.amp(micSensitivitySlider.value() * 20); // Adjust mic input sensitivity
        micLevel = mic.getLevel(); // Get the current mic level

        textSize(24);
        textAlign(LEFT, BOTTOM);
        text(`Mic Level: ${nf(micLevel, 0, 6)}`, 20, height - 20);
    }

    // Display water flow status
    if (waterFlowDetected) {
        fill(0, 150, 0);
        textSize(40);
        text("Water Flow Detected!", width / 2 - 150, height / 2);
    } else {
        fill(150, 0, 0);
        textSize(40);
        text("No Water Flow Detected", width / 2 - 180, height / 2);
    }
}
