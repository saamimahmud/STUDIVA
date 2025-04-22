import React, { useState, useEffect } from 'react';

// Simple placeholder for now
function AdminPanel() {
    const [questionsData, setQuestionsData] = useState(null); // { subject: [questions] }
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // Function to fetch all questions
    const fetchAdminQuestions = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('http://127.0.0.1:5000/admin/questions');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setQuestionsData(data);
        } catch (e) {
            setError(`Failed to load questions: ${e.message}`);
            console.error("Failed to fetch admin questions:", e);
        } finally {
            setIsLoading(false);
        }
    };

    // Fetch questions on component mount
    useEffect(() => {
        fetchAdminQuestions();
    }, []);

    // --- Handlers for Add/Edit/Delete (To be implemented) ---
    const handleAddQuestion = (subject, questionData) => {
        console.log("TODO: Add question to", subject, questionData);
        // Call POST /admin/questions/<subject>
        // Then call fetchAdminQuestions() to refresh
    };

    const handleEditQuestion = (subject, index, questionData) => {
        console.log("TODO: Edit question", subject, index, questionData);
        // Call PUT /admin/questions/<subject>/<index>
        // Then call fetchAdminQuestions() to refresh
    };

    const handleDeleteQuestion = (subject, index) => {
        console.log("TODO: Delete question", subject, index);
        // Call DELETE /admin/questions/<subject>/<index>
        // Then call fetchAdminQuestions() to refresh
    };
    
    // --- Render Logic --- 
    return (
        <div>
            <h2>Admin Panel - Manage Viva Questions</h2>

            {isLoading && <p>Loading questions...</p>}
            {error && <p style={{ color: 'red' }}>Error: {error}</p>}

            {questionsData && Object.keys(questionsData).length > 0 ? (
                Object.entries(questionsData).map(([subject, questions]) => (
                    <div key={subject} style={{ marginBottom: '2rem', border: '1px solid #eee', padding: '1rem' }}>
                        <h3>Subject: {subject}</h3>
                        {questions.length > 0 ? (
                            <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
                                {questions.map((q, index) => (
                                    <li key={index} style={{ borderBottom: '1px dashed #eee', marginBottom: '1rem', paddingBottom: '1rem' }}>
                                        <p><strong>Q{index + 1}:</strong> {q.question}</p>
                                        <p><strong>Expected:</strong> {q.expected}</p>
                                        {q.keywords && q.keywords.length > 0 && (
                                            <p><strong>Keywords:</strong> {q.keywords.join(', ')}</p>
                                        )}
                                        {/* TODO: Add Edit/Delete Buttons Here */}
                                        <button onClick={() => alert('Edit not implemented yet')} style={{ marginRight: '0.5rem' }}>Edit</button>
                                        <button onClick={() => alert('Delete not implemented yet')}>Delete</button>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p>No questions found for this subject.</p>
                        )}
                        {/* TODO: Add "Add Question" Button Here */}
                         <button onClick={() => alert('Add not implemented yet')}>Add New Question to {subject}</button>
                    </div>
                ))
            ) : (
                !isLoading && <p>No subjects or questions found.</p>
            )}
             {/* TODO: Add "Add New Subject" Functionality */} 
        </div>
    );
}

export default AdminPanel; 