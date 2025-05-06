function generateText() {
  // Make a call to the random endpoint to get a real example from your dataset
  fetch('http://127.0.0.1:5000/random')
    .then(response => response.json())
    .then(data => {
      document.getElementById("newsInput").value = data.text;
    })
    .catch(error => {
      console.error('Error fetching random text:', error);
      document.getElementById("newsInput").value = "Error fetching random text. The server may not be running.";
    });
}

function predict() {
  const text = document.getElementById("newsInput").value;
  
  if (!text.trim()) {
    document.getElementById("predictionResult").textContent = "Please enter some text first.";
    return;
  }
  
  // Show loading state
  document.getElementById("predictionResult").textContent = "Predicting...";
  
  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    // Store the full response for the details view
    window.lastPrediction = data;
    
    // Check if we have a prediction property in the response
    if (data.prediction !== undefined) {
      const resultClass = data.label === 0 ? "fake" : "real";
      const confidenceText = data.confidence ? 
        ` (Confidence: ${(data.confidence * 100).toFixed(2)}%)` : '';
      
      document.getElementById("predictionResult").innerHTML = 
        `<span class="prediction-${resultClass}">${data.prediction}${confidenceText}</span>`;
        
      // Also fetch feature importance to have it ready when user clicks details
      fetchFeatureImportance(text);
    } else {
      document.getElementById("predictionResult").textContent = "Error: Unexpected response format";
      console.error('Unexpected response format:', data);
    }
  })
  .catch(error => {
    console.error('Error:', error);
    document.getElementById("predictionResult").textContent = "Error predicting. Check console for details.";
  });
}

function fetchFeatureImportance(text) {
  fetch('http://127.0.0.1:5000/feature_importance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
  .then(response => response.json())
  .then(data => {
    window.featureImportance = data;
  })
  .catch(error => {
    console.error('Error fetching feature importance:', error);
  });
}

function submitFeedback(isCorrect) {
  if (!window.lastPrediction) {
    alert("Please make a prediction first before providing feedback.");
    return;
  }
  
  const text = document.getElementById("newsInput").value;
  const predictedLabel = window.lastPrediction.label;
  const correctLabel = isCorrect ? predictedLabel : (predictedLabel === 0 ? 1 : 0);
  
  fetch('http://127.0.0.1:5000/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      text: text,
      predicted_label: predictedLabel,
      correct_label: correctLabel
    })
  })
  .then(response => response.json())
  .then(data => {
    alert("Thank you for your feedback! This helps improve our models.");
  })
  .catch(error => {
    console.error('Error submitting feedback:', error);
    alert("There was an error submitting your feedback.");
  });
}

function showDetails() {
  const text = document.getElementById("newsInput").value;
  
  if (!text.trim()) {
    alert("Please enter some text and make a prediction first.");
    return;
  }
  
  // Get the current prediction result
  const predictionResult = document.getElementById("predictionResult").textContent;
  if (predictionResult === "Predicting..." || predictionResult === "Please enter some text first.") {
    alert("Please wait for the prediction to complete first.");
    return;
  }
  
  if (!window.lastPrediction) {
    alert("Please make a prediction first.");
    return;
  }
  
  const data = window.lastPrediction;
  const isPredictionFake = data.label === 0;
  
  // Create a detailed analysis as text (simple popup version)
  let details = "Detailed Analysis:\n\n";
  details += `Prediction: ${isPredictionFake ? 'FAKE NEWS' : 'REAL NEWS'}\n\n`;
  
  // Add confidence score
  const confidencePercent = data.confidence ? (data.confidence * 100).toFixed(2) : 'N/A';
  details += `Confidence Score: ${confidencePercent}%\n\n`;
  
  // Add key indicators
  details += "Key Indicators:\n";
  if (isPredictionFake) {
    details += "• Sensationalist language detected\n";
    details += "• Lack of credible sources\n";
    details += "• Emotional manipulation patterns\n";
    details += "• Inconsistent narrative structure\n";
  } else {
    details += "• Balanced reporting style\n";
    details += "• Credible source patterns identified\n";
    details += "• Factual presentation\n";
    details += "• Consistent narrative structure\n";
  }
  
  // Add text statistics
  const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
  details += `\nText Length: ${text.length} characters\n`;
  details += `Word Count: ${wordCount} words`;
  
  // Display as a simple JavaScript alert box
  alert(details);
  
  // Provide a way to submit feedback after seeing the details
  const feedbackResponse = confirm("Would you like to provide feedback on this prediction?\nClick OK if correct, Cancel if incorrect.");
  if (feedbackResponse !== null) {
    submitFeedback(feedbackResponse);
  }
}

// Keep these helper functions for other modal operations
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  modal.style.display = "block";
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  modal.style.display = "none";
}

// Close modal when clicking outside
window.onclick = function(event) {
  const modals = document.getElementsByClassName("modal");
  for (let modal of modals) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
}

// Handle form submissions
document.addEventListener('DOMContentLoaded', function() {
  // Add CSS styles for prediction results
  const style = document.createElement('style');
  style.innerHTML = `
    .prediction-fake {
      color: #d9534f;
      font-weight: bold;
    }
    .prediction-real {
      color: #5cb85c;
      font-weight: bold;
    }
  `;
  document.head.appendChild(style);
  
  // Initialize window.lastPrediction and featureImportance
  window.lastPrediction = null;
  window.featureImportance = null;
  
  const signupForm = document.getElementById('signupForm');
  if (signupForm) {
    signupForm.addEventListener('submit', function(e) {
      e.preventDefault();
      alert('Sign up functionality would be implemented here with a backend.');
      closeModal('signupModal');
    });
  }
  
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', function(e) {
      e.preventDefault();
      alert('Login functionality would be implemented here with a backend.');
      closeModal('loginModal');
    });
  }
});
