import React, { useState } from 'react';
import axios from 'axios'; // Import axios
import { ReactMediaRecorder } from "react-media-recorder"; // Import recorder

const scenarios = [
  "Ask a doubt about a concept taught in class.",
  "Explain why you came late to class.",
  "Politely disagree with a peer's point during a discussion.",
  "Respond to constructive feedback from the teacher about your assignment.",
  "Ask the teacher for an extension on a deadline."
];

function CommPracticePanel() {
  const [selectedScenario, setSelectedScenario] = useState(scenarios[0]);
  const [response, setResponse] = useState('');
  const [commAudioURL, setCommAudioURL] = useState(null); // Add state for audio URL
  const [evaluation, setEvaluation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false); // Add state for transcription loading
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setEvaluation(null);

    try {
      const apiResponse = await fetch('http://127.0.0.1:5000/evaluate_communication', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          scenario: selectedScenario, 
          response: response 
        })
      });

      if (!apiResponse.ok) {
        // Try to get error message from backend response if possible
        let errorMsg = `HTTP error! status: ${apiResponse.status}`;
        try {
            const errorData = await apiResponse.json();
            errorMsg = errorData.error || errorMsg;
        } catch (jsonError) {
            // Ignore if response is not JSON
        }
        throw new Error(errorMsg);
      }

      const evaluationData = await apiResponse.json();
      setEvaluation(evaluationData);

    } catch (err) {
      setError(`Evaluation failed: ${err.message}`);
      console.error("Evaluation error:", err);
      // Optionally show alert to user
      // alert(`Evaluation failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // --- Handle Audio Transcription ---
  const handleCommAudioUpload = async () => {
    if (!commAudioURL) return;
    setIsTranscribing(true);
    setError(null);
    const blob = await fetch(commAudioURL).then(r => r.blob());
    const formData = new FormData();
    formData.append('audio', blob, 'comm_recording.wav');

    try {
      const response = await axios.post('http://127.0.0.1:5000/transcribe', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      const transcript = response.data.transcript;
      setResponse(transcript); // Set the transcribed text as the response
    } catch (err) {
      const errorMsg = `Transcription failed: ${err.response?.data?.error || err.message}`;
      setError(errorMsg);
      alert(errorMsg);
    } finally {
      setIsTranscribing(false);
    }
  };

  return (
    <div>
      <h2>Classroom Communication Practice</h2>
      <p>Select a scenario and practice your response. Focus on politeness, clarity, and appropriateness.</p>

      <div style={{ marginBottom: '1rem' }}>
        <label htmlFor="scenario-select">Select Scenario:</label><br />
        <select 
          id="scenario-select"
          value={selectedScenario}
          onChange={e => {
              setSelectedScenario(e.target.value);
              setResponse(''); // Clear response when scenario changes
              setCommAudioURL(null);
              setEvaluation(null);
              setError(null);
          }}
          style={{ width: '100%', padding: '0.5rem' }}
          disabled={isLoading || isTranscribing}
        >
          {scenarios.map((scenario, index) => (
            <option key={index} value={scenario}>{scenario}</option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <label htmlFor="response-input">Your Response (Type or Record):</label><br />
        <textarea
          id="response-input"
          rows="6"
          cols="70"
          value={response}
          onChange={e => setResponse(e.target.value)}
          placeholder="Type your response here or use voice input below..."
          style={{ width: '100%', padding: '0.5rem' }}
          disabled={isLoading || isTranscribing}
        />
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <ReactMediaRecorder
          audio
          onStop={(blobUrl) => setCommAudioURL(blobUrl)} // Use commAudioURL state
          render={({ status, startRecording, stopRecording }) => (
            <div style={{ marginBottom: '0.5rem' }}>
              <button onClick={startRecording} disabled={status === "recording" || isLoading || isTranscribing}>Start Recording</button>
              <button onClick={stopRecording} disabled={status !== "recording" || isLoading || isTranscribing} style={{ marginLeft: '1rem' }}>Stop Recording</button>
              <span style={{ marginLeft: '1rem' }}>Status: {status}</span>
            </div>
          )}
        />
      </div>

      {commAudioURL && (
        <div style={{ marginBottom: '1rem' }}>
          <audio controls src={commAudioURL} style={{ verticalAlign: 'middle' }}></audio>
          <button onClick={handleCommAudioUpload} disabled={!commAudioURL || isLoading || isTranscribing} style={{ marginLeft: '1rem' }}>
            {isTranscribing ? 'Transcribing...' : 'Transcribe & Use Response'}
          </button>
        </div>
      )}

      <button onClick={handleSubmit} disabled={isLoading || isTranscribing || !response}>
        {isLoading ? 'Evaluating...' : 'Submit Response for Evaluation'}
      </button>

      {error && <p style={{ color: 'red', marginTop: '1rem' }}>Error: {error}</p>}

      {evaluation && (
        <div style={{ marginTop: '2rem', borderTop: '1px solid #ccc', paddingTop: '1rem' }}>
          <h3>Evaluation Results:</h3>
          <p><strong>Tone:</strong> {evaluation.tone || 'N/A'}</p>
          <p><strong>Politeness:</strong> {evaluation.politeness}</p>
          <p><strong>Clarity:</strong> {evaluation.clarity}</p>
          <p><strong>Grammar:</strong> {evaluation.grammar}</p>
          <p><strong>Appropriateness:</strong> {evaluation.appropriateness}</p>
          <p><strong>Feedback & Suggestions:</strong></p>
          <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', fontSize: 'inherit', background: '#f8f8f8', padding: '10px', borderRadius: '4px' }}>
            {evaluation.feedback}
          </pre>
        </div>
      )}
    </div>
  );
}

export default CommPracticePanel; 