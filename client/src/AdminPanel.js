import React, { useState, useEffect } from 'react';

// Component for the Add Question Form
const AddQuestionForm = ({ subject, onAdd, onCancel }) => {
    const [newQuestion, setNewQuestion] = useState('');
    const [newExpected, setNewExpected] = useState('');
    const [newKeywords, setNewKeywords] = useState(''); // Stored as comma-separated string
    const [isAdding, setIsAdding] = useState(false);
    const [addError, setAddError] = useState(null);

    const handleAddSubmit = async (e) => {
        e.preventDefault();
        setIsAdding(true);
        setAddError(null);
        const keywordsArray = newKeywords.split(',').map(k => k.trim()).filter(k => k);
        try {
            await onAdd(subject, { 
                question: newQuestion, 
                expected: newExpected, 
                keywords: keywordsArray 
            });
            // Clear form on success (handled by parent potentially closing form)
            setNewQuestion('');
            setNewExpected('');
            setNewKeywords('');
            // onCancel(); // Call cancel to hide form after adding
        } catch (error) {
            setAddError(error.message);
        } finally {
            setIsAdding(false);
        }
    };

    return (
        <form onSubmit={handleAddSubmit} style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #ccc', backgroundColor: '#f9f9f9' }}>
            <h4>Add New Question to {subject}</h4>
            <div style={{ marginBottom: '0.5rem' }}>
                <label>Question:</label><br />
                <textarea 
                    value={newQuestion} 
                    onChange={e => setNewQuestion(e.target.value)} 
                    required 
                    style={{ width: '95%', minHeight: '40px' }} 
                />
            </div>
            <div style={{ marginBottom: '0.5rem' }}>
                <label>Expected Answer:</label><br />
                <textarea 
                    value={newExpected} 
                    onChange={e => setNewExpected(e.target.value)} 
                    required 
                    style={{ width: '95%', minHeight: '40px' }} 
                />
            </div>
            <div style={{ marginBottom: '1rem' }}>
                <label>Keywords (comma-separated):</label><br />
                <input 
                    type="text" 
                    value={newKeywords} 
                    onChange={e => setNewKeywords(e.target.value)} 
                    style={{ width: '95%' }} 
                />
            </div>
            <button type="submit" disabled={isAdding} style={{ marginRight: '0.5rem' }}>
                {isAdding ? 'Adding...' : 'Save New Question'}
            </button>
            <button type="button" onClick={onCancel} disabled={isAdding}>
                Cancel
            </button>
            {addError && <p style={{ color: 'red', marginTop: '0.5rem' }}>Error: {addError}</p>}
        </form>
    );
};

function AdminPanel() {
    const [questionsData, setQuestionsData] = useState(null); // { subject: [questions] }
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    // State to track which subject's add form is open
    const [addingToSubject, setAddingToSubject] = useState(null); 

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

    // --- Implement Add Question Handler ---
    const handleAddQuestion = async (subject, questionData) => {
        console.log("Adding question to", subject, questionData);
        // Set loading/error states specific to add operation? Maybe within form component.
        setError(null); // Clear general error
        try {
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subject}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(questionData)
            });
            const result = await response.json(); // Always try to parse JSON
            if (!response.ok) {
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            console.log("Add success:", result);
            setAddingToSubject(null); // Close the form
            await fetchAdminQuestions(); // Refresh the list
            // Optionally: show success message
            alert("Question added successfully!");
        } catch (e) {
            console.error("Failed to add question:", e);
            setError(`Failed to add question: ${e.message}`); // Show error
            // Re-throw error for the form component to catch if needed
            throw e; 
        }
    };

    // --- Implement Delete Question Handler ---
    const handleDeleteQuestion = async (subject, index) => {
        // Confirmation dialog
        const qText = questionsData[subject][index]?.question || `Question ${index + 1}`;
        if (!window.confirm(`Are you sure you want to delete this question from ${subject}?\n\n"${qText}"`)) {
            return; // Abort if user cancels
        }

        console.log("Deleting question", subject, index);
        setError(null); // Clear previous errors
        // Consider adding a specific loading state for the item being deleted

        try {
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subject}/${index}`, {
                method: 'DELETE',
            });
            const result = await response.json(); // Try to parse JSON, might contain error
            if (!response.ok) {
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            console.log("Delete success:", result);
            await fetchAdminQuestions(); // Refresh the list
            alert("Question deleted successfully!"); 
        } catch (e) {
            console.error("Failed to delete question:", e);
            setError(`Failed to delete question: ${e.message}`); // Show error
        }
    };

    // --- Handlers for Add/Edit/Delete (To be implemented) ---
    const handleEditQuestion = (subject, index, questionData) => {
        console.log("TODO: Edit question", subject, index, questionData);
        // Call PUT /admin/questions/<subject>/<index>
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
                                        <button onClick={() => alert('Edit not implemented yet')} style={{ marginRight: '0.5rem' }}>Edit</button>
                                        <button onClick={() => handleDeleteQuestion(subject, index)}>Delete</button>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p>No questions found for this subject.</p>
                        )}
                        
                        {/* Toggle Add Question Form */} 
                        {addingToSubject === subject ? (
                            <AddQuestionForm 
                                subject={subject} 
                                onAdd={handleAddQuestion} 
                                onCancel={() => setAddingToSubject(null)} 
                            />
                        ) : (
                            <button onClick={() => setAddingToSubject(subject)}>
                                Add New Question to {subject}
                            </button>
                        )}
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