import React, { useState, useEffect } from 'react';

// Add at the top with other imports
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Helper component to render individual result details (similar to App.js)
const SessionResultDetail = ({ result, index }) => (
  <div key={index} style={{ borderBottom: '1px solid #eee', paddingBottom: '15px', marginBottom: '15px' }}>
    <h4><strong>Q{index + 1}: {result.question}</strong></h4>
    <p><strong>Your Answer:</strong> {result.answer}</p>
    <p><strong>Expected:</strong> {result.expected}</p>
    <p>
      <strong>Similarity:</strong>
      <span style={{ color: (result.similarity_score || 0) > 60 ? 'green' : 'red', marginLeft: '5px' }}>
        {(result.similarity_score || 0).toFixed(2)}%
      </span>
    </p>
    <p>
      <strong>Keywords:</strong>
      <span style={{ color: (result.keyword_score || 0) > 60 ? 'green' : 'red', marginLeft: '5px' }}>
        {(result.keyword_score || 0).toFixed(2)}%
      </span>
    </p>
    <p>
      <strong>Fluency:</strong>
      <span style={{ marginLeft: '5px' }}>
        {(result.fluency_score || 0).toFixed(2)}
      </span>
    </p>
    <p><strong>Feedback:</strong> {result.feedback}</p>
    <p><strong>Confidence:</strong>
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
);

function SavedSessionsViewer() {
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSessionId, setSelectedSessionId] = useState(null); // State for viewing details

  useEffect(() => {
    const fetchSessions = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_URL}/sessions`);
        if (!response.ok) {
          let errorMsg = `HTTP error! status: ${response.status}`;
          try {
            const errorData = await response.json();
            errorMsg = errorData.error || errorMsg;
          } catch (jsonError) { /* Ignore */ }
          throw new Error(errorMsg);
        }
        const data = await response.json();
        setSessions(data);
      } catch (e) {
        setError(`Failed to load saved sessions: ${e.message}`);
        console.error("Failed to fetch sessions:", e);
        setSessions([]); // Ensure sessions is empty on error
      } finally {
        setIsLoading(false);
      }
    };

    fetchSessions();
  }, []); // Fetch only once on mount

  // Find the selected session object
  const selectedSession = selectedSessionId 
    ? sessions.find(session => session.id === selectedSessionId)
    : null;

  return (
    <div>
      {/* Show Details View if a session is selected */} 
      {selectedSession ? (
        <div>
            <button onClick={() => setSelectedSessionId(null)} style={{ marginBottom: '1rem' }}>
                &larr; Back to List
            </button>
            <h2>Session Details (ID: {selectedSession.id})</h2>
            <p><strong>Student:</strong> {selectedSession.studentName || 'Unknown'}</p>
            <p><strong>Subject:</strong> {selectedSession.subject || 'N/A'}</p>
            <p><strong>Mode:</strong> {selectedSession.mode || 'N/A'}</p>
            <p><strong>Saved At:</strong> {selectedSession.createdAt ? new Date(selectedSession.createdAt).toLocaleString() : 'N/A'}</p>
            {/* TODO: Add Student Name here later */} 
            
            <h3>Summary:</h3>
            <p>
                Avg Similarity: {selectedSession.summary?.averageSimilarityScore || 'N/A'}% | 
                Avg Keywords: {selectedSession.summary?.averageKeywordScore || 'N/A'}% | 
                Total Questions: {selectedSession.summary?.totalQuestions || 'N/A'}
            </p>

            <h3>Individual Results:</h3>
            {selectedSession.results && selectedSession.results.length > 0 ? (
                selectedSession.results.map((result, index) => (
                    <SessionResultDetail key={index} result={result} index={index} />
                ))
            ) : (
                <p>No individual results found for this session.</p>
            )}
        </div>
      ) : (
        // Show List View otherwise
        <div> 
            <h2>Saved Viva Sessions</h2>
            {isLoading && <p>Loading saved sessions...</p>}
            {error && <p style={{ color: 'red' }}>Error: {error}</p>}
            {!isLoading && !error && sessions.length === 0 && (
                <p>No saved sessions found.</p>
            )}
            {!isLoading && !error && sessions.length > 0 && (
                <ul style={{ listStyle: 'none', padding: 0 }}>
                {sessions.map((session) => (
                    <li key={session.id} style={{ border: '1px solid #ccc', marginBottom: '1rem', padding: '1rem' }}>
                    <strong>Student:</strong> {session.studentName || 'Unknown'}<br />
                    <strong>Subject:</strong> {session.subject || 'N/A'}<br />
                     {/* TODO: Add Student Name here later */} 
                    <strong>Saved At:</strong> {session.createdAt ? new Date(session.createdAt).toLocaleString() : 'N/A'}<br />
                    <strong>Avg Similarity:</strong> {session.summary?.averageSimilarityScore || 'N/A'}%<br />
                    <strong>Avg Keywords:</strong> {session.summary?.averageKeywordScore || 'N/A'}%<br />
                    <strong>Questions:</strong> {session.summary?.totalQuestions || 'N/A'}<br />
                    <button onClick={() => setSelectedSessionId(session.id)} style={{ marginTop: '0.5rem' }}>
                        View Details
                    </button>
                    </li>
                ))}
                </ul>
            )}
        </div>
      )}
    </div>
  );
}

export default SavedSessionsViewer; 