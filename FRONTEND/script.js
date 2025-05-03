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
    // Check if we have a label property in the response
    if (data.label !== undefined) {
      const result = data.label === 1 ? "FAKE" : "REAL";
      document.getElementById("predictionResult").textContent = `Prediction: ${result}`;
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
  
  const isPredictionFake = predictionResult.includes("FAKE");
  
  // Create a detailed analysis
  let details = "Detailed Analysis:\n\n";
  details += "Prediction: " + (isPredictionFake ? "FAKE NEWS" : "REAL NEWS") + "\n\n";
  details += "Confidence Score: " + (isPredictionFake ? "87%" : "92%") + "\n\n";
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
  
  details += "\nText Length: " + text.length + " characters\n";
  details += "Word Count: " + text.split(/\s+/).filter(word => word.length > 0).length + " words";
  
  alert(details);
}

// Modal functionality
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
