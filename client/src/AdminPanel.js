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

// --- Component for the Edit Question Form ---
const EditQuestionForm = ({ subject, index, questionData, onSave, onCancel }) => {
    const [editedQuestion, setEditedQuestion] = useState(questionData.question);
    const [editedExpected, setEditedExpected] = useState(questionData.expected);
    const [editedKeywords, setEditedKeywords] = useState((questionData.keywords || []).join(', '));
    const [isSaving, setIsSaving] = useState(false);
    const [editError, setEditError] = useState(null);

    const handleSaveSubmit = async (e) => {
        e.preventDefault();
        setIsSaving(true);
        setEditError(null);
        const keywordsArray = editedKeywords.split(',').map(k => k.trim()).filter(k => k);
        try {
            await onSave(subject, index, { 
                question: editedQuestion, 
                expected: editedExpected, 
                keywords: keywordsArray 
            });
            // Parent component will handle closing the form by resetting editing state
        } catch (error) {
            setEditError(error.message);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <form onSubmit={handleSaveSubmit} style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #4CAF50', backgroundColor: '#f0fff0' }}>
            <h4>Editing Question {index + 1} in {subject}</h4>
            <div style={{ marginBottom: '0.5rem' }}>
                <label>Question:</label><br />
                <textarea 
                    value={editedQuestion} 
                    onChange={e => setEditedQuestion(e.target.value)} 
                    required 
                    style={{ width: '95%', minHeight: '40px' }} 
                />
            </div>
            <div style={{ marginBottom: '0.5rem' }}>
                <label>Expected Answer:</label><br />
                <textarea 
                    value={editedExpected} 
                    onChange={e => setEditedExpected(e.target.value)} 
                    required 
                    style={{ width: '95%', minHeight: '40px' }} 
                />
            </div>
            <div style={{ marginBottom: '1rem' }}>
                <label>Keywords (comma-separated):</label><br />
                <input 
                    type="text" 
                    value={editedKeywords} 
                    onChange={e => setEditedKeywords(e.target.value)} 
                    style={{ width: '95%' }} 
                />
            </div>
            <button type="submit" disabled={isSaving} style={{ marginRight: '0.5rem' }}>
                {isSaving ? 'Saving...' : 'Save Changes'}
            </button>
            <button type="button" onClick={onCancel} disabled={isSaving}>
                Cancel
            </button>
            {editError && <p style={{ color: 'red', marginTop: '0.5rem' }}>Error: {editError}</p>}
        </form>
    );
};

// --- Component for Managing Scenarios ---
const ScenarioManager = ({ scenariosData, isLoadingScenarios, scenarioError, onAdd, onEdit, onDelete }) => {
    const [editingScenario, setEditingScenario] = useState(null); // { index: number, text: string } | null
    const [addingScenario, setAddingScenario] = useState(false);
    const [newScenarioText, setNewScenarioText] = useState('');

    // Edit Form Submit Handler
    const handleSaveEditScenario = (e, index, updatedText) => {
        e.preventDefault();
        if (!updatedText.trim()) {
             alert("Scenario text cannot be empty.");
             return;
        } 
        if (updatedText === scenariosData[index]) {
            setEditingScenario(null); // Cancel if no change
            return;
        }
        onEdit(index, updatedText.trim()); // Call parent handler
        // Parent handler will close form on success
    };

    // Add Form Submit Handler
    const handleAddScenarioSubmit = (e) => {
        e.preventDefault();
        if (!newScenarioText.trim()) {
            alert("Scenario text cannot be empty.");
            return;
        }
        onAdd(newScenarioText.trim()); // Call parent handler
        setNewScenarioText(''); // Clear local form state
        // Parent handler will close form on success
    };

    return (
        <div style={{ marginTop: '2rem', borderTop: '2px solid #ccc', paddingTop: '1rem' }}>
            <h3>Manage Communication Scenarios</h3>
            {isLoadingScenarios && <p>Loading scenarios...</p>}
            {scenarioError && <p style={{ color: 'red' }}>Error: {scenarioError}</p>}
            
            {!isLoadingScenarios && scenariosData.length > 0 && (
                <ul style={{ listStyle: 'decimal', paddingLeft: '20px' }}>
                    {scenariosData.map((scenario, index) => (
                        <li key={index} style={{ marginBottom: '1rem' }}>
                            {editingScenario && editingScenario.index === index ? (
                                // Edit Form (inline)
                                <form onSubmit={(e) => handleSaveEditScenario(e, index, editingScenario.text)} style={{ display: 'inline' }}>
                                    <textarea 
                                        value={editingScenario.text}
                                        onChange={(e) => setEditingScenario({...editingScenario, text: e.target.value})}
                                        rows={2}
                                        required
                                        style={{ width: '70%', marginRight: '10px', verticalAlign: 'middle' }}
                                    />
                                    {/* Add loading state for save button if needed */}
                                    <button type="submit" style={{ marginRight: '5px' }}>Save</button> 
                                    <button type="button" onClick={() => setEditingScenario(null)}>Cancel</button>
                                </form>
                            ) : (
                                // Display View
                                <>
                                    <span style={{ marginRight: '15px' }}>{scenario}</span>
                                    <button 
                                        onClick={() => setEditingScenario({ index, text: scenario })} 
                                        style={{ marginRight: '5px' }} 
                                        disabled={addingScenario || editingScenario !== null}
                                    >Edit</button>
                                    <button 
                                        onClick={() => onDelete(index)} 
                                        disabled={addingScenario || editingScenario !== null}
                                    >Delete</button>
                                </>
                            )}
                        </li>
                    ))}
                </ul>
            )}
            {!isLoadingScenarios && scenariosData.length === 0 && <p>No scenarios found.</p>}
            
            {/* Add Scenario Form Toggle/Area */} 
            {addingScenario ? (
                <form onSubmit={handleAddScenarioSubmit} style={{ marginTop: '1rem' }}>
                    <textarea 
                        value={newScenarioText}
                        onChange={e => setNewScenarioText(e.target.value)}
                        placeholder="Enter new scenario text..."
                        rows={2}
                        required
                        style={{ width: '70%', marginRight: '10px', verticalAlign: 'middle' }}
                    />
                    {/* Add loading state for add button if needed */}
                    <button type="submit" style={{ marginRight: '5px' }}>Add Scenario</button> 
                    <button type="button" onClick={() => { setAddingScenario(false); setNewScenarioText(''); }}>Cancel</button>
                </form>
            ) : (
                <button onClick={() => setAddingScenario(true)} disabled={editingScenario !== null} style={{ marginTop: '1rem' }}>
                    Add New Scenario
                </button>
            )}
        </div>
    );
};

// --- Main Admin Panel Component --- 
function AdminPanel() {
    // --- State for Questions (existing) --- 
    const [questionsData, setQuestionsData] = useState(null);
    const [isLoadingQuestions, setIsLoadingQuestions] = useState(false); // Renamed from isLoading
    const [questionError, setQuestionError] = useState(null); // Renamed from error
    const [addingToSubject, setAddingToSubject] = useState(null); 
    const [editingQuestion, setEditingQuestion] = useState(null);

    // --- State for Scenarios --- 
    const [scenariosData, setScenariosData] = useState([]);
    const [isLoadingScenarios, setIsLoadingScenarios] = useState(false);
    const [scenarioError, setScenarioError] = useState(null);
    
    // --- Combined Error for Operations ---
    const [generalError, setGeneralError] = useState(null);

    // --- Fetch Questions (existing, renamed state vars) ---
    const fetchAdminQuestions = async () => { 
        setIsLoadingQuestions(true);
        setQuestionError(null);
        try {
            const response = await fetch('http://127.0.0.1:5000/admin/questions');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setQuestionsData(data);
        } catch (e) {
            setQuestionError(`Failed to load questions: ${e.message}`);
            console.error("Failed to fetch admin questions:", e);
        } finally {
            setIsLoadingQuestions(false);
        }
     };

    // --- Fetch Scenarios --- 
    const fetchAdminScenarios = async () => {
        setIsLoadingScenarios(true);
        setScenarioError(null);
        try {
            const response = await fetch('http://127.0.0.1:5000/admin/scenarios');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setScenariosData(data);
        } catch (e) {
            setScenarioError(`Failed to load scenarios: ${e.message}`);
            console.error("Failed to fetch scenarios:", e);
        } finally {
            setIsLoadingScenarios(false);
        }
    };

    // Fetch all data on component mount
    useEffect(() => {
        fetchAdminQuestions();
        fetchAdminScenarios(); // Fetch scenarios too
    }, []);

    // --- Question Handlers (existing, updated error state) ---
    const handleAddQuestion = async (subject, questionData) => {
        setGeneralError(null); 
        try {
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subject}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(questionData)
            });
            const result = await response.json(); 
            if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
            setAddingToSubject(null);
            await fetchAdminQuestions(); 
            alert("Question added successfully!");
        } catch (e) {
            console.error("Failed to add question:", e);
            setGeneralError(`Failed to add question: ${e.message}`); // Use general error
            throw e; 
        }
    };
    const handleDeleteQuestion = async (subject, index) => {
        const qText = questionsData[subject][index]?.question || `Question ${index + 1}`;
        if (!window.confirm(`Are you sure you want to delete this question from ${subject}?\n\n"${qText}"`)) return;
        setGeneralError(null);
        try {
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subject}/${index}`, { method: 'DELETE' });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
            await fetchAdminQuestions();
            alert("Question deleted successfully!"); 
        } catch (e) {
            console.error("Failed to delete question:", e);
            setGeneralError(`Failed to delete question: ${e.message}`); // Use general error
        }
    };
    const handleEditQuestion = async (subject, index, updatedData) => {
        setGeneralError(null);
        try {
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subject}/${index}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updatedData)
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
            setEditingQuestion(null); 
            await fetchAdminQuestions(); 
            alert("Question updated successfully!");
        } catch (e) {
            console.error("Failed to update question:", e);
            setGeneralError(`Failed to update question: ${e.message}`); // Use general error
            throw e; 
        }
    };

    // --- Scenario Handlers (To be implemented) ---
    const handleAddScenario = async (scenarioText) => { 
        setGeneralError(null);
        // Maybe add specific loading state to ScenarioManager?
        try {
            const response = await fetch('http://127.0.0.1:5000/admin/scenarios', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenarioText })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
            // ScenarioManager form state handles clearing/closing via props
            await fetchAdminScenarios(); // Refresh list
            alert("Scenario added successfully!");
        } catch (e) {
            console.error("Failed to add scenario:", e);
            setGeneralError(`Failed to add scenario: ${e.message}`);
            // Re-throw if ScenarioManager needs to handle its own loading/error
            // throw e;
        } 
     };
    const handleEditScenario = async (index, scenarioText) => { 
        setGeneralError(null);
        try {
            const response = await fetch(`http://127.0.0.1:5000/admin/scenarios/${index}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenarioText })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
            // ScenarioManager form state handles closing via props
            await fetchAdminScenarios(); // Refresh list
            alert("Scenario updated successfully!");
        } catch (e) {
            console.error("Failed to update scenario:", e);
            setGeneralError(`Failed to update scenario: ${e.message}`);
            // Re-throw if ScenarioManager needs to handle its own loading/error
            // throw e;
        } 
    };
    const handleDeleteScenario = async (index) => { 
        const scenarioText = scenariosData[index] || `Scenario ${index + 1}`;
        if (!window.confirm(`Are you sure you want to delete this scenario?\n\n"${scenarioText}"`)) return;
        setGeneralError(null);
        try {
            const response = await fetch(`http://127.0.0.1:5000/admin/scenarios/${index}`, { method: 'DELETE' });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
            await fetchAdminScenarios(); // Refresh list
            alert("Scenario deleted successfully!");
        } catch (e) {
            console.error("Failed to delete scenario:", e);
            setGeneralError(`Failed to delete scenario: ${e.message}`);
        }
    };

    // --- Render Logic (To be updated) ---
    return (
        <div>
            <h2>Admin Panel - Manage Viva Questions</h2>

            {isLoadingQuestions && <p>Loading questions...</p>}
            {questionError && <p style={{ color: 'red' }}>Error: {questionError}</p>}

            {questionsData && Object.keys(questionsData).length > 0 ? (
                Object.entries(questionsData).map(([subject, questions]) => (
                    <div key={subject} style={{ marginBottom: '2rem', border: '1px solid #eee', padding: '1rem' }}>
                        <h3>Subject: {subject}</h3>
                        {questions.length > 0 ? (
                            <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
                                {questions.map((q, index) => (
                                    <li key={index} style={{ borderBottom: '1px dashed #eee', marginBottom: '1rem', paddingBottom: '1rem' }}>
                                        {
                                            // Check if this question is being edited
                                            editingQuestion && editingQuestion.subject === subject && editingQuestion.index === index 
                                            ? (
                                                // Render Edit Form
                                                <EditQuestionForm 
                                                    subject={subject} 
                                                    index={index} 
                                                    questionData={q} 
                                                    onSave={handleEditQuestion} 
                                                    onCancel={() => setEditingQuestion(null)} 
                                                />
                                            ) 
                                            : (
                                                // Render Normal View
                                                <>
                                                    <p><strong>Q{index + 1}:</strong> {q.question}</p>
                                                    <p><strong>Expected:</strong> {q.expected}</p>
                                                    {q.keywords && q.keywords.length > 0 && (
                                                        <p><strong>Keywords:</strong> {q.keywords.join(', ')}</p>
                                                    )}
                                                    {/* Update Edit Button onClick */}
                                                    <button 
                                                        onClick={() => setEditingQuestion({ subject, index })} 
                                                        disabled={editingQuestion !== null || addingToSubject !== null} // Disable if any form is open
                                                        style={{ marginRight: '0.5rem' }}
                                                    >
                                                        Edit
                                                    </button>
                                                    <button 
                                                        onClick={() => handleDeleteQuestion(subject, index)}
                                                        disabled={editingQuestion !== null || addingToSubject !== null} // Disable if any form is open
                                                    >
                                                        Delete
                                                    </button>
                                                </>
                                            )
                                        }
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p>No questions found for this subject.</p>
                        )}
                        
                        {/* Toggle Add Question Form (disable button if editing) */} 
                        {addingToSubject === subject ? (
                            <AddQuestionForm 
                                subject={subject} 
                                onAdd={handleAddQuestion} 
                                onCancel={() => setAddingToSubject(null)} 
                            />
                        ) : (
                            <button 
                                onClick={() => setAddingToSubject(subject)}
                                disabled={editingQuestion !== null || addingToSubject !== null} // Disable if any form is open
                            >
                                Add New Question to {subject}
                            </button>
                        )}
                    </div>
                ))
            ) : (
                !isLoadingQuestions && <p>No subjects or questions found.</p>
            )}
            
            {/* Scenario Management Section */}
            <ScenarioManager 
                scenariosData={scenariosData}
                isLoadingScenarios={isLoadingScenarios}
                scenarioError={scenarioError}
                onAdd={handleAddScenario}
                onEdit={handleEditScenario}
                onDelete={handleDeleteScenario}
            />
        </div>
    );
}

export default AdminPanel; 