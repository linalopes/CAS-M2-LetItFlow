// Variables
let audioContext;
let model;
let microphone;
let sourceNode;

// Load TensorFlow.js model
async function loadModel() {
  model = await tf.loadGraphModel('rnn_model_tfjs/model.json');
  console.log("Model loaded");
}

// Start listening to the microphone
document.getElementById('start-btn').addEventListener('click', async () => {
  // Request access to the microphone
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  microphone = audioContext.createMediaStreamSource(stream);

  // Analyze the audio in real time
  analyzeAudio();
});

// Analyze audio stream
function analyzeAudio() {
  const analyser = audioContext.createAnalyser();
  microphone.connect(analyser);

  analyser.fftSize = 2048;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Float32Array(bufferLength);

  const mfccFeatures = []; // To accumulate MFCC frames (should match the shape 400x13)

  // Set an amplitude threshold for detecting valid sound (tune this value based on your needs)
  const amplitudeThreshold = 0.01;

  function processAudio() {
    analyser.getFloatTimeDomainData(dataArray);

    // Calculate the root mean square (RMS) to estimate signal strength
    const rms = Math.sqrt(dataArray.reduce((sum, value) => sum + value * value, 0) / dataArray.length);

    // Skip processing if the signal is too weak (below threshold)
    if (rms < amplitudeThreshold) {
      document.getElementById('result').innerText = "No significant sound detected, please pour water.";
      requestAnimationFrame(processAudio);
      return;
    }

    // Extract MFCCs using Meyda
    const mfcc = Meyda.extract('mfcc', dataArray);

    // Check if MFCCs are valid and non-empty
    if (mfcc && mfcc.length === 13) {
      mfccFeatures.push(mfcc);
    } else {
      document.getElementById('result').innerText = "No valid sound detected, please try again.";
      requestAnimationFrame(processAudio);
      return;
    }

    // If we have enough MFCC frames (e.g., 400), make a prediction
    if (mfccFeatures.length >= 400) {
      // Convert MFCC features to a tensor of shape [1, 400, 13]
      const inputTensor = tf.tensor([mfccFeatures.slice(0, 400)], [1, 400, 13]);

      // Run the model only if valid input data exists
      model.executeAsync({ 'keras_tensor_3': inputTensor }).then(prediction => {
        const predictionData = prediction[0].dataSync();
        const predictedIndex = predictionData.indexOf(Math.max(...predictionData));

        // Convert prediction to label
        const labels = ['short', 'medium', 'long'];
        const resultLabel = labels[predictedIndex];

        // Display result and play sound
        document.getElementById('result').innerText = `Detected: ${resultLabel}`;
        playChord(resultLabel);

        // Reset the MFCC feature buffer
        mfccFeatures.length = 0;
      }).catch(error => {
        console.error("Error running the model:", error);
      });

      // Clear features for the next prediction cycle
      mfccFeatures.length = 0;
    }

    requestAnimationFrame(processAudio);
  }

  processAudio();
}





// Play a sound based on the result
function playChord(label) {
  const audio = document.getElementById('audio-output');
  
  let audioFile;
  if (label === 'short') {
    audioFile = 'path/to/short_chord.mp3';
  } else if (label === 'medium') {
    audioFile = 'path/to/medium_chord.mp3';
  } else if (label === 'long') {
    audioFile = 'path/to/long_chord.mp3';
  }

  audio.src = audioFile;
  audio.play();
}

// Load the model when the page loads
window.onload = () => {
  loadModel();
};
