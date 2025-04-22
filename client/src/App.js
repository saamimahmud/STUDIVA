import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ReactMediaRecorder } from "react-media-recorder";
import './App.css';
// Import the new component
import CommPracticePanel from './CommPracticePanel';
// Import Admin Panel
import AdminPanel from './AdminPanel';

// Simple placeholder component for Admin Mode
// const AdminPanelPlaceholder = () => (
//   <div>
//     <h2>Admin Panel</h2>
//     <p>Manage Viva questions and Communication scenarios.</p>
//     <p><i>(Admin Panel UI under construction)</i></p>
//   </div>
// );

function App() {
  // --- State for Mode Switching ---
  const [currentMode, setCurrentMode] = useState('viva'); // 'viva', 'communication', or 'admin'

  // --- State for Viva Mode ---
  const [studentAnswer, setStudentAnswer] = useState('');
  const [audioURL, setAudioURL] = useState(null);
  // Note: feedback and confidence are now part of 'results'
  const [questionIndex, setQuestionIndex] = useState(0);
  const [selectedSubject, setSelectedSubject] = useState('');
  const [questionMode, setQuestionMode] = useState('');
  const [questions, setQuestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);
  const [vivaCompleted, setVivaCompleted] = useState(false);

  // --- State for Communication Mode (Add as needed) ---
  // This state will likely live within CommPracticePanel itself now

  // --- useEffect for Viva Mode (fetching questions) ---
  useEffect(() => {
    // Only fetch questions if in viva mode
    if (currentMode === 'viva' && selectedSubject && questionMode) {
      const fetchQuestions = async () => {
        setIsLoading(true);
        setError(null);
        try {
          const response = await fetch(`http://127.0.0.1:5000/questions?subject=${selectedSubject}&mode=${questionMode}`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          setQuestions(data);
          setQuestionIndex(0);
          setResults([]);
          setVivaCompleted(false);
          setStudentAnswer('');
          setAudioURL(null);
        } catch (e) {
          setError(`Failed to load questions: ${e.message}`);
          setQuestions([]);
        } finally {
          setIsLoading(false);
        }
      };
      fetchQuestions();
    } else {
      // Reset viva state if subject/mode changes or mode switches
      setQuestions([]);
      setResults([]);
      setVivaCompleted(false);
    }
  }, [currentMode, selectedSubject, questionMode]); // Add currentMode dependency

  // --- Handlers for Viva Mode ---
  const handleSubmit = async () => {
    if (!questions[questionIndex]) return;

    const currentQuestion = questions[questionIndex];

    try {
      const response = await fetch('http://127.0.0.1:5000/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          answer: studentAnswer,
          expected: currentQuestion.expected,
          keywords: currentQuestion.keywords || []
        })
      });

      if (!response.ok) {
        throw new Error(`Evaluation failed: ${response.statusText}`);
      }

      const data = await response.json();
      const result = {
        question: currentQuestion.question,
        expected: currentQuestion.expected,
        answer: studentAnswer,
        similarity_score: data.similarity_score,
        keyword_score: data.keyword_score,
        fluency_score: data.fluency_score,
        feedback: data.feedback,
        confidence: data.confidence
      };

      setResults([...results, result]);
      setStudentAnswer('');
      setAudioURL(null);

      if (questionIndex + 1 < questions.length) {
        setQuestionIndex(questionIndex + 1);
      } else {
        setVivaCompleted(true);
      }
    } catch (err) {
      setError(`Error submitting answer: ${err.message}`);
      alert(`Error submitting answer: ${err.message}`);
    }
  };

  const handleSaveSession = async () => {
    if (results.length === 0) {
      alert("No results to save.");
      return;
    }
    try {
      const totalSimilarity = results.reduce((acc, item) => acc + (item.similarity_score || 0), 0);
      const totalKeywords = results.reduce((acc, item) => acc + (item.keyword_score || 0), 0);
      const totalFluency = results.reduce((acc, item) => acc + (item.fluency_score || 0), 0);

      const avgSimilarity = results.length > 0 ? totalSimilarity / results.length : 0;
      const avgKeywords = results.length > 0 ? totalKeywords / results.length : 0;
      const avgFluency = results.length > 0 ? totalFluency / results.length : 0;

      const response = await fetch('http://127.0.0.1:5000/save_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject: selectedSubject,
          mode: questionMode,
          results: results,
          summary: {
            averageSimilarityScore: avgSimilarity.toFixed(2),
            averageKeywordScore: avgKeywords.toFixed(2),
            averageFluencyScore: avgFluency.toFixed(2),
            totalQuestions: results.length
          }
        }),
      });

      const data = await response.json();
      alert("Session saved as: " + data.filename);
    } catch (err) {
      alert("Error saving session: " + err.message);
    }
  };

  const handleAudioUpload = async () => {
    if (!audioURL) return;
    const blob = await fetch(audioURL).then(r => r.blob());
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');

    try {
      const response = await axios.post('http://127.0.0.1:5000/transcribe', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      const transcript = response.data.transcript;
      setStudentAnswer(transcript);
    } catch (err) {
      setError(`Transcription failed: ${err.response?.data?.error || err.message}`);
      alert(`Transcription failed: ${err.response?.data?.error || err.message}`);
    }
  };

  // --- Render Logic ---
  const renderVivaPanel = () => (
    <>
      {/* Subject Selector */}
      <div style={{ marginBottom: '1rem' }}>
        <label>Select Subject:</label><br />
        <select value={selectedSubject} onChange={(e) => setSelectedSubject(e.target.value)} disabled={isLoading}>
          <option value="">-- Select Subject --</option>
          <option value="AI">AI</option>
          <option value="CN">Computer Networks</option>
          <option value="DBMS">DBMS</option>
        </select>
      </div>

      {/* Mode Selector */}
      <div style={{ marginBottom: '2rem' }}>
        <label>Select Question Mode:</label><br />
        <select value={questionMode} onChange={(e) => setQuestionMode(e.target.value)} disabled={isLoading || !selectedSubject}>
          <option value="">-- Select Mode --</option>
          <option value="sequential">Sequential</option>
          <option value="random">Random</option>
        </select>
      </div>

      {/* Loading/Error/Start Message */}
      {isLoading && <p>Loading questions...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {!isLoading && !error && (!selectedSubject || !questionMode) && (
        <p>Please select a subject and question mode to start.</p>
      )}

      {/* Viva Interface */}
      {!vivaCompleted && !isLoading && !error && questions.length > 0 && questions[questionIndex] ? (
        <>
          <h2>Question {questionIndex + 1} of {questions.length}:</h2>
          <p><strong>{questions[questionIndex].question}</strong></p>

          {/* Answer Input */}
          <div style={{ marginBottom: '1rem' }}>
            <label>Your Answer:</label><br />
            <textarea
              rows="4"
              cols="60"
              value={studentAnswer}
              onChange={e => setStudentAnswer(e.target.value)}
              placeholder="Type your answer here or use voice input..."
            />
          </div>

          {/* Audio Recorder */}
          <ReactMediaRecorder
            audio
            onStop={(blobUrl) => setAudioURL(blobUrl)}
            render={({ status, startRecording, stopRecording }) => (
              <div style={{ marginBottom: '1rem' }}>
                <button onClick={startRecording} disabled={status === "recording"}>Start Recording</button>
                <button onClick={stopRecording} disabled={status !== "recording"} style={{ marginLeft: '1rem' }}>Stop Recording</button>
                <span style={{ marginLeft: '1rem' }}>Status: {status}</span>
              </div>
            )}
          />

          {/* Audio Player & Transcribe Button */}
          {audioURL && (
            <div style={{ marginBottom: '1rem' }}>
              <audio controls src={audioURL} style={{ verticalAlign: 'middle' }}></audio>
              <button onClick={handleAudioUpload} disabled={!audioURL} style={{ marginLeft: '1rem' }}>Transcribe & Use</button>
            </div>
          )}

          {/* Submit Button */}
          <button onClick={handleSubmit} disabled={!studentAnswer}>Submit Answer</button>
        </>
      ) : vivaCompleted ? (
        // Results Display
        <div>
          <h2>Viva Completed!</h2>
          <h3>Results:</h3>
          {results.map((result, index) => (
            <div key={index} style={{ borderBottom: '1px solid #ccc', paddingBottom: '15px', marginBottom: '20px' }}>
              <h4><strong>Q{index + 1}: {result.question}</strong></h4>
              <p><strong>Your Answer:</strong> {result.answer}</p>
              <p><strong>Expected:</strong> {result.expected}</p>
              <p>
                <strong>Similarity Score:</strong>
                <span style={{ color: (result.similarity_score || 0) > 60 ? 'green' : 'red', marginLeft: '5px' }}>
                  {(result.similarity_score || 0).toFixed(2)}%
                </span>
              </p>
              <p>
                <strong>Keyword Score:</strong>
                <span style={{ color: (result.keyword_score || 0) > 60 ? 'green' : 'red', marginLeft: '5px' }}>
                  {(result.keyword_score || 0).toFixed(2)}%
                </span>
              </p>
              <p>
                <strong>Fluency Score (Readability):</strong>
                <span style={{ marginLeft: '5px' }}>
                  {(result.fluency_score || 0).toFixed(2)}
                </span>
              </p>
              <p><strong>Feedback:</strong> {result.feedback}</p>
              <p><strong>Confidence Level:</strong>
                <span style={{
                  backgroundColor: result.confidence === "High" ? "#d4edda" :
                                   result.confidence === "Medium" ? "#fff3cd" :
                                   "#f8d7da",
                  padding: '2px 8px',
                  borderRadius: '6px',
                  marginLeft: '6px',
                  fontWeight: 'bold'
                }}>
                  {result.confidence}
                </span>
              </p>
            </div>
          ))}
          {/* Save Session Button */}
          <button onClick={handleSaveSession} style={{ marginTop: '1rem' }}>
            Save Session
          </button>
        </div>
      ) : null}
    </>
  );

  return (
    <div className="App" style={{ padding: '2rem', fontFamily: 'Arial' }}>
      {/* Mode Switching Buttons */}
      <div style={{ marginBottom: '2rem', borderBottom: '1px solid #ccc', paddingBottom: '1rem' }}>
        <button
          onClick={() => setCurrentMode('viva')}
          disabled={currentMode === 'viva'}
          style={{ marginRight: '1rem', padding: '0.5rem 1rem' }}
        >
          Viva Practice
        </button>
        <button
          onClick={() => setCurrentMode('communication')}
          disabled={currentMode === 'communication'}
          style={{ marginRight: '1rem', padding: '0.5rem 1rem' }}
        >
          Communication Practice
        </button>
        {/* Add Admin Mode Button */}
        <button
          onClick={() => setCurrentMode('admin')}
          disabled={currentMode === 'admin'}
          style={{ padding: '0.5rem 1rem' }}
        >
          Admin Mode
        </button>
      </div>

      {/* Render content based on current mode */}
      {currentMode === 'viva' && (
        <div>
          <h1>AI Viva Evaluator</h1>
          {renderVivaPanel()}
        </div>
      )}

      {currentMode === 'communication' && (
        <div>
          {/* Render Communication Panel */}
          <CommPracticePanel />
        </div>
      )}

      {/* Render Admin Mode */}
      {currentMode === 'admin' && (
        <div>
           {/* Use the actual AdminPanel component */}
          <AdminPanel /> 
        </div>
      )}

    </div>
  );
}

export default App;
