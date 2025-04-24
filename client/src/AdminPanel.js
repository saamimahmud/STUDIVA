import React, { useState, useEffect } from 'react';
import { useAuth } from './context/AuthContext'; // Import useAuth

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import IconButton from '@mui/material/IconButton';
import Divider from '@mui/material/Divider';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Chip from '@mui/material/Chip'; // For keywords

// MUI Icons
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';

// Component for the Add Question Form
const AddQuestionForm = ({ subjectId, subjectName, onAdd, onCancel }) => {
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
            await onAdd(subjectId, { 
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
        <Box component="form" onSubmit={handleAddSubmit} sx={{ mt: 2, p: 2, border: '1px dashed grey' }}>
            <Typography variant="h6" gutterBottom>Add New Question to {subjectName}</Typography>
            {addError && <Alert severity="error" sx={{ mb: 2 }}>{addError}</Alert>}
            <TextField
                label="Question"
                value={newQuestion}
                onChange={e => setNewQuestion(e.target.value)}
                required
                multiline
                rows={2}
                fullWidth
                margin="normal"
            />
            <TextField
                label="Expected Answer"
                value={newExpected}
                onChange={e => setNewExpected(e.target.value)}
                required
                multiline
                rows={2}
                fullWidth
                margin="normal"
            />
            <TextField
                label="Keywords (comma-separated)"
                value={newKeywords}
                onChange={e => setNewKeywords(e.target.value)}
                fullWidth
                margin="normal"
            />
            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button type="submit" variant="contained" disabled={isAdding} startIcon={<SaveIcon />}>
                    {isAdding ? 'Adding...' : 'Save New Question'}
                </Button>
                <Button type="button" variant="outlined" onClick={onCancel} disabled={isAdding} startIcon={<CancelIcon />}>
                    Cancel
                </Button>
            </Box>
        </Box>
    );
};

// --- Component for the Edit Question Form ---
const EditQuestionForm = ({ subjectId, subjectName, questionId, questionData, onSave, onCancel }) => {
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
        
        // Construct data object WITHOUT the id
        const updatedData = { 
            question: editedQuestion,
            expected: editedExpected,
            keywords: keywordsArray
        };

        try {
             // Call onSave with subjectId, questionId, and the updated data
            await onSave(subjectId, questionId, updatedData);
            // Parent (AdminPanel) handles closing via setEditingQuestion(null) after successful refresh
            // No need to call onCancel() here on success
        } catch (error) {
            // Display error from parent
            setEditError(error.message);
            // Keep form open on error
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <Box component="form" onSubmit={handleSaveSubmit} sx={{ mt: 2, p: 2, border: '1px solid green', backgroundColor: '#f0fff0' }}>
            <Typography variant="h6" gutterBottom>Editing Question in {subjectName}</Typography>
            {editError && <Alert severity="error" sx={{ mb: 2 }}>{editError}</Alert>}
            <TextField
                label="Question"
                value={editedQuestion}
                onChange={e => setEditedQuestion(e.target.value)}
                required
                multiline
                rows={2}
                fullWidth
                margin="normal"
            />
            <TextField
                label="Expected Answer"
                value={editedExpected}
                onChange={e => setEditedExpected(e.target.value)}
                required
                multiline
                rows={2}
                fullWidth
                margin="normal"
            />
            <TextField
                label="Keywords (comma-separated)"
                value={editedKeywords}
                onChange={e => setEditedKeywords(e.target.value)}
                fullWidth
                margin="normal"
            />
            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button type="submit" variant="contained" color="success" disabled={isSaving} startIcon={<SaveIcon />}>
                    {isSaving ? 'Saving...' : 'Save Changes'}
                </Button>
                <Button type="button" variant="outlined" onClick={onCancel} disabled={isSaving} startIcon={<CancelIcon />}>
                    Cancel
                </Button>
            </Box>
        </Box>
    );
};

// --- Component for Managing Scenarios ---
const ScenarioManager = ({ scenariosData, isLoadingScenarios, scenarioError, onAdd, onEdit, onDelete, currentUser }) => {
    // Update state to hold ID instead of index
    const [editingScenario, setEditingScenario] = useState(null); // { id: string, text: string } | null
    const [addingScenario, setAddingScenario] = useState(false);
    const [newScenarioText, setNewScenarioText] = useState('');
    const [isSavingEdit, setIsSavingEdit] = useState(false);
    const [isSavingAdd, setIsSavingAdd] = useState(false);
    const [formError, setFormError] = useState(null);

    // Edit Form Submit Handler - use ID
    const handleSaveEditScenario = async (e, id, updatedText) => {
        e.preventDefault();
        if (!updatedText.trim()) {
             alert("Scenario text cannot be empty.");
             return;
        } 
        
        try {
            // Assume onEdit is an async function (returns a promise)
            await onEdit(id, updatedText.trim()); // Call parent handler with ID and wait
            // If onEdit succeeds (doesn't throw), close the form
            setEditingScenario(null); 
        } catch (error) {
            // Error is likely handled/displayed by the parent (AdminPanel)
            // We could potentially show an error message here too if needed
            console.error("Edit failed in ScenarioManager:", error); 
            // Don't close the form if the edit failed
        }
    };

    // Add Form Submit Handler (unchanged)
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
        <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>Manage Communication Scenarios</Typography>
            {isLoadingScenarios && <Box sx={{display: 'flex', justifyContent: 'center', p:2}}><CircularProgress /></Box>}
            {scenarioError && <Alert severity="error">Error loading scenarios: {scenarioError}</Alert>}
            
            {!isLoadingScenarios && scenariosData.length > 0 && (
                <List>
                    {scenariosData.map((scenario) => (
                        <ListItem 
                            key={scenario.id} 
                            divider
                            secondaryAction={
                                <Box sx={{display: 'flex', gap: 0.5}}>
                                    <IconButton 
                                        edge="end" 
                                        aria-label="edit"
                                        onClick={() => setEditingScenario({ id: scenario.id, text: scenario.scenario })}
                                        disabled={addingScenario || editingScenario?.id === scenario.id || isSavingEdit}
                                    >
                                        <EditIcon />
                                    </IconButton>
                                    <IconButton 
                                        edge="end" 
                                        aria-label="delete"
                                        onClick={() => onDelete(scenario.id)}
                                        disabled={addingScenario || !!editingScenario || isSavingAdd}
                                    >
                                        <DeleteIcon />
                                    </IconButton>
                                </Box>
                            }
                        >
                            {editingScenario?.id === scenario.id ? (
                                <Box component="form" onSubmit={(e) => handleSaveEditScenario(e, editingScenario.id, editingScenario.text)} sx={{ width: '100%', pt: 1, pb: 1 }}>
                                    <TextField 
                                        value={editingScenario.text}
                                        onChange={(e) => setEditingScenario({...editingScenario, text: e.target.value})}
                                        multiline rows={2} required fullWidth variant="outlined" size="small"
                                        sx={{ mb: 1 }}
                                    />
                                     {formError && <Alert severity="warning" size="small" sx={{ mb: 1 }}>{formError}</Alert>}
                                    <Box sx={{display: 'flex', gap: 1}}>
                                        <Button type="submit" size="small" variant="contained" disabled={isSavingEdit} startIcon={<SaveIcon />}>
                                            {isSavingEdit ? 'Saving...' : 'Save'}
                                        </Button>
                                        <Button type="button" size="small" variant="outlined" onClick={() => { setEditingScenario(null); setFormError(null); }} startIcon={<CancelIcon />}>
                                            Cancel
                                        </Button>
                                    </Box>
                                </Box>
                            ) : (
                                <ListItemText primary={scenario.scenario} />
                            )}
                        </ListItem>
                    ))}
                </List>
            )}
            {!isLoadingScenarios && scenariosData.length === 0 && !scenarioError && (
                <Typography sx={{mt: 2}}>No scenarios found.</Typography>
            )}
            
            {/* Add Scenario Form Toggle/Area */}
            <Box sx={{ mt: 3 }}>
                {addingScenario ? (
                    <Box component="form" onSubmit={handleAddScenarioSubmit}>
                         <Typography variant="subtitle1" gutterBottom>Add New Scenario</Typography>
                          {formError && <Alert severity="warning" sx={{ mb: 1 }}>{formError}</Alert>}
                         <TextField 
                            value={newScenarioText}
                            onChange={e => setNewScenarioText(e.target.value)}
                            placeholder="Enter new scenario text..."
                            multiline rows={2} required fullWidth variant="outlined"
                            sx={{ mb: 1 }}
                        />
                        <Box sx={{display: 'flex', gap: 1}}>
                            <Button type="submit" variant="contained" disabled={isSavingAdd} startIcon={<AddIcon />}>
                                {isSavingAdd ? 'Adding...' : 'Add Scenario'}
                            </Button>
                            <Button type="button" variant="outlined" onClick={() => { setAddingScenario(false); setNewScenarioText(''); setFormError(null); }} startIcon={<CancelIcon />}>
                                Cancel
                            </Button>
                         </Box>
                    </Box>
                ) : (
                    <Button 
                        onClick={() => setAddingScenario(true)} 
                        variant="contained"
                        disabled={!!editingScenario}
                        startIcon={<AddIcon />}
                    >
                        Add New Scenario
                    </Button>
                )}
            </Box>
        </Paper>
    );
};

// --- Main Admin Panel Component --- 
function AdminPanel() {
    // State for managing questions
    const [questionsData, setQuestionsData] = useState([]);
    const [isLoadingQuestions, setIsLoadingQuestions] = useState(true);
    const [questionError, setQuestionError] = useState(null);
    const [editingQuestion, setEditingQuestion] = useState(null); // { subjectId: string, questionId: string } | null
    const [addingToSubject, setAddingToSubject] = useState(null); // subjectId | null

    // State for managing scenarios
    const [scenariosData, setScenariosData] = useState([]);
    const [isLoadingScenarios, setIsLoadingScenarios] = useState(true);
    const [scenarioError, setScenarioError] = useState(null);
    // Editing/adding state is now inside ScenarioManager

    // State for adding new subjects
    const [isAddingSubject, setIsAddingSubject] = useState(false);
    const [newSubjectName, setNewSubjectName] = useState('');
    const [addSubjectError, setAddSubjectError] = useState(null);
    
    // General Error State
    const [generalError, setGeneralError] = useState(null);

    // State for Tabs
    const [activeTab, setActiveTab] = useState(0); // 0 for Viva, 1 for Scenarios

    // State for controlling Accordion expansion
    const [expandedAccordionId, setExpandedAccordionId] = useState(false); // Stores the ID of the expanded accordion, or false

    const { currentUser } = useAuth(); // Get current user from context

    // --- Handlers for Tabs ---
    const handleTabChange = (event, newValue) => {
        setActiveTab(newValue);
    };

    // --- Handler for Accordion Change ---
    const handleAccordionChange = (subjectId) => (event, isExpanded) => {
        setExpandedAccordionId(isExpanded ? subjectId : false);
        // Also ensure that if we manually collapse an accordion, we clear any associated adding/editing state within it
        if (!isExpanded) {
            if (addingToSubject === subjectId) {
                setAddingToSubject(null);
            }
            if (editingQuestion?.subjectId === subjectId) {
                setEditingQuestion(null);
            }
        }
    };

    // --- Fetch Functions (getAuthToken, fetchAdminQuestions, fetchAdminScenarios) ---
    const getAuthToken = async (user) => {
        if (!user) {
            throw new Error("User not logged in.");
        }
        try {
            // Force refresh parameter (true) can be used if needed, but usually not required
            // unless you've just updated claims and need them immediately.
            const token = await user.getIdToken(/* forceRefresh */ false);
            return token;
        } catch (error) {
            console.error("Error getting auth token:", error);
            throw new Error("Could not retrieve authentication token.");
        }
    };

    const fetchAdminQuestions = async () => { 
        setIsLoadingQuestions(true);
        setQuestionError(null);
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch('http://127.0.0.1:5000/admin/questions', {
                headers: { 'Authorization': `Bearer ${token}` } // Add token
            });
            if (!response.ok) {
                // Try parsing error message if available
                let errorMsg = `HTTP error! status: ${response.status}`;
                try { const errData = await response.json(); errorMsg = errData.error || errorMsg; } catch(e){}
                throw new Error(errorMsg);
            }
            const data = await response.json();
            setQuestionsData(data);
        } catch (e) {
            setQuestionError(`Failed to load questions: ${e.message}`);
            console.error("Failed to fetch admin questions:", e);
            setQuestionsData([]); 
        } finally {
            setIsLoadingQuestions(false);
        }
     };

    const fetchAdminScenarios = async () => {
        setIsLoadingScenarios(true);
        setScenarioError(null);
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch('http://127.0.0.1:5000/admin/scenarios', {
                 headers: { 'Authorization': `Bearer ${token}` } // Add token
            });
            if (!response.ok) {
                 // Try parsing error message if available
                let errorMsg = `HTTP error! status: ${response.status}`;
                try { const errData = await response.json(); errorMsg = errData.error || errorMsg; } catch(e){}
                throw new Error(errorMsg);
            }
            const data = await response.json();
            setScenariosData(data);
        } catch (e) {
            setScenarioError(`Failed to load scenarios: ${e.message}`);
            console.error("Failed to fetch scenarios:", e);
            setScenariosData([]);
        } finally {
            setIsLoadingScenarios(false);
        }
    };

    // --- UseEffect Hooks ---
    useEffect(() => {
        // Ensure currentUser is available before fetching protected data
        if (currentUser) { 
            fetchAdminQuestions();
            fetchAdminScenarios();
        }
        // If currentUser becomes null (logout), maybe clear admin data?
        // else {
        //     setQuestionsData(null);
        //     setScenariosData([]);
        // }
    }, [currentUser]); // Re-run if currentUser changes

    useEffect(() => {
        // Ensure currentUser is available before fetching protected data
        if (currentUser) { 
            fetchAdminQuestions();
            fetchAdminScenarios();
        }
        // If currentUser becomes null (logout), maybe clear admin data?
        // else {
        //     setQuestionsData(null);
        //     setScenariosData([]);
        // }
    }, [currentUser]); // Re-run if currentUser changes

    // --- Viva Question Handlers (handleAddQuestion, handleDeleteQuestion, handleEditQuestion) ---
    const handleAddQuestion = async (subjectId, questionData) => {
        setGeneralError(null); 
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subjectId}`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json', 
                    'Authorization': `Bearer ${token}` // Add token
                },
                body: JSON.stringify(questionData)
            });
            const result = await response.json(); 
            if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
            setAddingToSubject(null);
            await fetchAdminQuestions(); 
            alert("Question added successfully!");
        } catch (e) {
            console.error("Failed to add question:", e);
            setGeneralError(`Failed to add question: ${e.message}`);
            // Don't re-throw here, let the form handle its state
        }
    };
    const handleDeleteQuestion = async (subjectId, questionId) => {
        const subjectObj = questionsData?.find(subj => subj.id === subjectId);
        const questionObj = subjectObj?.questions.find(q => q.id === questionId);
        const qText = questionObj?.question || `Question ID: ${questionId}`;
        const subjectName = subjectObj?.subject || subjectId;
        
        if (!window.confirm(`Are you sure you want to delete this question from ${subjectName}?\n\n"${qText}"`)) return;
        setGeneralError(null);
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subjectId}/${questionId}`, { 
                method: 'DELETE', 
                headers: { 'Authorization': `Bearer ${token}` } // Add token
             });
            
            if (!response.ok) {
                const result = await response.json().catch(() => ({}));
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            await fetchAdminQuestions();
            alert("Question deleted successfully!"); 
        } catch (e) {
            console.error("Failed to delete question:", e);
            setGeneralError(`Failed to delete question: ${e.message}`);
        }
    };
    const handleEditQuestion = async (subjectId, questionId, updatedData) => { 
        setGeneralError(null);
        // We expect updatedData to be { question, expected, keywords }
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch(`http://127.0.0.1:5000/admin/questions/${subjectId}/${questionId}`, {
                method: 'PUT',
                headers: { 
                    'Content-Type': 'application/json',
                     'Authorization': `Bearer ${token}` // Add token
                },
                body: JSON.stringify(updatedData) 
            });
            
             if (!response.ok) {
                const result = await response.json().catch(() => ({}));
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            //const updatedQuestion = await response.json(); // Get updated question back
            
            setEditingQuestion(null); // Close form
            await fetchAdminQuestions(); // Refresh list
            alert("Question updated successfully!");
            // We don't need to return anything here for EditQuestionForm 
        } catch (e) {
            console.error("Failed to update question:", e);
            setGeneralError(`Failed to update question: ${e.message}`);
            throw e; // Re-throw for EditQuestionForm handler
        }
    };

    // --- Scenario Handlers (handleAddScenario, handleEditScenario, handleDeleteScenario) ---
    const handleAddScenario = async (scenarioText) => { 
        setGeneralError(null);
        // Maybe add specific loading state to ScenarioManager?
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch('http://127.0.0.1:5000/admin/scenarios', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json', 
                    'Authorization': `Bearer ${token}` // Add token
                },
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
     
    // Updated handleEditScenario to use scenarioId
    const handleEditScenario = async (scenarioId, scenarioText) => { 
        setGeneralError(null);
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch(`http://127.0.0.1:5000/admin/scenarios/${scenarioId}`, {
                method: 'PUT',
                headers: { 
                    'Content-Type': 'application/json', 
                    'Authorization': `Bearer ${token}` // Add token
                },
                body: JSON.stringify({ scenario: scenarioText })
            });
            // Check status, parse JSON only if needed
            if (!response.ok) {
                const result = await response.json().catch(() => ({})); 
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            // Don't need result data for success usually
            await fetchAdminScenarios(); // Refresh list
            alert("Scenario updated successfully!");
            // We are now relying on ScenarioManager's handleSaveEditScenario to call its own setEditingScenario(null)
        } catch (e) {
            console.error("Failed to update scenario:", e);
            setGeneralError(`Failed to update scenario: ${e.message}`);
            // Rethrow the error so ScenarioManager's catch block is triggered
            throw e; 
        } 
    };
    const handleDeleteScenario = async (scenarioId) => { 
        if (!window.confirm(`Are you sure you want to delete scenario ID: ${scenarioId}?`)) return;
        setGeneralError(null);
        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch(`http://127.0.0.1:5000/admin/scenarios/${scenarioId}`, { 
                method: 'DELETE',
                 headers: { 'Authorization': `Bearer ${token}` } // Add token
            });
            if (!response.ok) { 
                const result = await response.json().catch(() => ({}));
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            await fetchAdminScenarios();
            alert("Scenario deleted successfully!");
        } catch (e) {
            console.error("Failed to delete scenario:", e);
            setGeneralError(`Failed to delete scenario: ${e.message}`);
        }
    };

    // --- Subject Handlers ---
    const handleAddNewSubject = async (e) => {
        e.preventDefault(); 
        if (!newSubjectName.trim()) {
            setAddSubjectError("Subject name cannot be empty.");
            return;
        }
        setAddSubjectError(null);
        setGeneralError(null);
        // TODO: Add loading state indication if desired

        try {
            const token = await getAuthToken(currentUser); // Get token
            const response = await fetch('http://127.0.0.1:5000/admin/subjects', { 
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json', 
                    'Authorization': `Bearer ${token}` // Add token
                },
                body: JSON.stringify({ subject: newSubjectName.trim() })
            });
            const result = await response.json();
            if (!response.ok) {
                 // Use the error message from the backend if available
                throw new Error(result.error || `HTTP error! status: ${response.status}`)
            };
            
            // Success:
            setIsAddingSubject(false); // Close form
            setNewSubjectName(''); 
            await fetchAdminQuestions(); // Refresh the questions/subjects list
            alert(`Subject '${result.subject}' added successfully!`); // Use name from response

        } catch (err) {
            console.error("Failed to add subject:", err);
            // Display the specific error message from the backend or fetch
            setAddSubjectError(err.message); 
            // Optionally use generalError as well/instead
            // setGeneralError(`Failed to add subject: ${err.message}`);
        } finally {
            // TODO: Remove loading state indication if added
        }
    };

    // --- Render Logic --- 
    return (
        <Box sx={{ width: '100%' }}>
            {/* Use Box for overall container and padding */}
            <Typography variant="h4" component="h1" gutterBottom>Admin Panel</Typography>

            {/* Tabs for switching views */} 
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                <Tabs value={activeTab} onChange={handleTabChange} aria-label="Admin Panel Tabs">
                    <Tab label="Viva Questions" id="admin-tab-0" aria-controls="admin-tabpanel-0" />
                    <Tab label="Communication Scenarios" id="admin-tab-1" aria-controls="admin-tabpanel-1" />
                </Tabs>
            </Box>

            {/* General Error Display */} 
            {generalError && (
                 <Alert severity="error" sx={{ mb: 2 }} onClose={() => setGeneralError(null)}>
                    Operation Error: {generalError}
                </Alert>
            )}

            {/* Tab Panel for Viva Questions */} 
            <div
                role="tabpanel"
                hidden={activeTab !== 0}
                id={`admin-tabpanel-0`}
                aria-labelledby={`admin-tab-0`}
            >
                {activeTab === 0 && (
                    <Box sx={{ p: { xs: 1, sm: 2 } }}> {/* Add some padding */}
                        {/* --- Content for Viva Questions (Refactored) --- */} 

                        <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3 }}>Manage Viva Questions</Typography>

                        {isLoadingQuestions && <Box sx={{display: 'flex', justifyContent: 'center', p: 3}}><CircularProgress /></Box>}
                        {questionError && <Alert severity="warning" sx={{mb: 2}}>{questionError}</Alert>}

                        {/* --- Add Subject Section --- */} 
                        <Box sx={{ mb: 3, pb: 2, borderBottom: 1, borderColor: 'divider' }}>
                            {!isAddingSubject ? (
                                <Button 
                                    variant="contained"
                                    startIcon={<AddIcon />}
                                    onClick={() => setIsAddingSubject(true)}
                                    disabled={editingQuestion !== null || addingToSubject !== null} // Disable if other forms are open
                                >
                                    Add New Subject
                                </Button>
                            ) : (
                                <Paper elevation={2} sx={{ p: 2, mt: 1 }}>
                                    <Box component="form" onSubmit={handleAddNewSubject}>
                                        <Typography variant="h6" gutterBottom>Create New Subject</Typography>
                                        {addSubjectError && <Alert severity="error" sx={{ mb: 2 }}>{addSubjectError}</Alert>}
                                        <TextField
                                            label="Subject Name"
                                            value={newSubjectName}
                                            onChange={e => setNewSubjectName(e.target.value)}
                                            placeholder="Enter new subject name"
                                            required
                                            fullWidth
                                            margin="normal"
                                            size="small"
                                        />
                                        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                                            <Button type="submit" variant="contained" startIcon={<SaveIcon />}>
                                                Save Subject
                                            </Button>
                                            <Button 
                                                type="button" 
                                                variant="outlined"
                                                onClick={() => { setIsAddingSubject(false); setNewSubjectName(''); setAddSubjectError(null); }} 
                                                startIcon={<CancelIcon />}
                                            >
                                                Cancel
                                            </Button>
                                        </Box>
                                    </Box>
                                </Paper>
                            )}
                        </Box>
                        {/* --- End Add Subject Section --- */} 

                        {/* --- Subject Accordions --- */} 
                        {questionsData && questionsData.length > 0 ? (
                            <Box>
                                {questionsData.map((subjectData) => {
                                    // Determine if this specific accordion should be expanded
                                    const isExpanded = 
                                        expandedAccordionId === subjectData.id || 
                                        addingToSubject === subjectData.id || 
                                        editingQuestion?.subjectId === subjectData.id;
                                        
                                    return (
                                        <Accordion 
                                            key={subjectData.id} 
                                            sx={{ mb: 1 }} 
                                            // Disable other accordions ONLY if an ADD or EDIT form is open inside *another* accordion
                                            disabled={(
                                                (!!editingQuestion && editingQuestion.subjectId !== subjectData.id) || 
                                                (!!addingToSubject && addingToSubject !== subjectData.id)
                                            )}
                                            // Control expansion based on state OR if adding/editing inside
                                            expanded={isExpanded}
                                            // Handle manual expansion/collapse clicks
                                            onChange={handleAccordionChange(subjectData.id)}
                                        >
                                            <AccordionSummary
                                                expandIcon={<ExpandMoreIcon />}
                                                aria-controls={`panel-${subjectData.id}-content`}
                                                id={`panel-${subjectData.id}-header`}
                                            >
                                                <Typography sx={{ flexShrink: 0, fontWeight: 'medium' }}>
                                                    {subjectData.subject}
                                                </Typography>
                                                <Typography sx={{ ml: 2, color: 'text.secondary' }}>
                                                    ({subjectData.questions?.length || 0} questions)
                                                </Typography>
                                            </AccordionSummary>
                                            <AccordionDetails sx={{ pt: 0 }}>
                                                {/* --- Question List for Subject --- */} 
                                                {subjectData.questions && subjectData.questions.length > 0 ? (
                                                    <List disablePadding>
                                                        {subjectData.questions.map((q, index) => (
                                                            <ListItem key={q.id} divider sx={{ display: 'block', pl: 0, pr: 0, py: 2 }}>
                                                                {
                                                                    editingQuestion && editingQuestion.subjectId === subjectData.id && editingQuestion.questionId === q.id 
                                                                    ? (
                                                                        // Display Edit Form
                                                                        <EditQuestionForm 
                                                                            subjectId={subjectData.id}
                                                                            subjectName={subjectData.subject} 
                                                                            questionId={q.id}
                                                                            questionData={q}
                                                                            onSave={handleEditQuestion} 
                                                                            onCancel={() => setEditingQuestion(null)} 
                                                                        />
                                                                    ) 
                                                                    : (
                                                                        // Display Question Details
                                                                        <Box>
                                                                             <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                                                                <Box sx={{ flexGrow: 1, mr: 1 }}>
                                                                                    <Typography variant="body1" gutterBottom><strong>Q{index + 1}:</strong> {q.question}</Typography>
                                                                                    <Typography variant="body2" color="text.secondary" gutterBottom><strong>Expected:</strong> {q.expected}</Typography>
                                                                                    {q.keywords && q.keywords.length > 0 && (
                                                                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                                                                                            <Typography variant="body2" sx={{ mr: 0.5, fontWeight: 'bold' }}>Keywords:</Typography>
                                                                                            {q.keywords.map((kw, i) => <Chip key={i} label={kw} size="small" />)}
                                                                                        </Box>
                                                                                    )}
                                                                                </Box>
                                                                                {/* Action Buttons */} 
                                                                                 <Box sx={{ display: 'flex', gap: 0.5 }}>
                                                                                    <IconButton 
                                                                                        size="small"
                                                                                        onClick={() => setEditingQuestion({ subjectId: subjectData.id, questionId: q.id })} 
                                                                                        disabled={editingQuestion !== null || addingToSubject !== null} 
                                                                                        aria-label="edit question"
                                                                                    >
                                                                                        <EditIcon fontSize="small"/>
                                                                                    </IconButton>
                                                                                    <IconButton 
                                                                                        size="small"
                                                                                        color="error"
                                                                                        onClick={() => handleDeleteQuestion(subjectData.id, q.id)}
                                                                                        disabled={editingQuestion !== null || addingToSubject !== null} 
                                                                                        aria-label="delete question"
                                                                                    >
                                                                                        <DeleteIcon fontSize="small"/>
                                                                                    </IconButton>
                                                                                </Box>
                                                                            </Box>
                                                                        </Box>
                                                                    )
                                                                }
                                                            </ListItem>
                                                        ))}
                                                    </List>
                                                ) : (
                                                    <Typography sx={{ mt: 2, fontStyle: 'italic', color: 'text.secondary' }}>No questions found for this subject.</Typography>
                                                )}
                                                {/* --- End Question List --- */} 

                                                {/* --- Add Question Form / Button --- */} 
                                                <Box sx={{ mt: 3 }}>
                                                    {addingToSubject === subjectData.id ? (
                                                        <AddQuestionForm 
                                                            subjectId={subjectData.id}
                                                            subjectName={subjectData.subject}
                                                            onAdd={handleAddQuestion} 
                                                            onCancel={() => setAddingToSubject(null)} 
                                                        />
                                                    ) : (
                                                        <Button 
                                                            variant="outlined"
                                                            size="small"
                                                            startIcon={<AddIcon />}
                                                            onClick={() => setAddingToSubject(subjectData.id)}
                                                            disabled={editingQuestion !== null || addingToSubject !== null}
                                                        >
                                                            Add New Question
                                                        </Button>
                                                    )}
                                                </Box>
                                                {/* --- End Add Question --- */} 
                                            </AccordionDetails>
                                        </Accordion>
                                    );
                                })}
                            </Box>
                        ) : (
                            !isLoadingQuestions && !questionError && <Typography sx={{mt: 3, fontStyle: 'italic', textAlign: 'center'}}>No subjects found. Use 'Add New Subject' to create one.</Typography>
                        )}
                        {/* --- End Subject Accordions --- */} 

                    </Box>
                )}
            </div>

             {/* Tab Panel for Communication Scenarios */} 
             <div
                role="tabpanel"
                hidden={activeTab !== 1}
                id={`admin-tabpanel-1`}
                aria-labelledby={`admin-tab-1`}
            >
                {activeTab === 1 && (
                    <Box sx={{ p: 3 }}>
                        <ScenarioManager 
                            scenariosData={scenariosData}
                            isLoadingScenarios={isLoadingScenarios}
                            scenarioError={scenarioError}
                            onAdd={handleAddScenario}
                            onEdit={handleEditScenario}
                            onDelete={handleDeleteScenario} 
                            currentUser={currentUser} // Pass currentUser if needed by ScenarioManager
                        />
                    </Box>
                )}
            </div>

        </Box>
    );
}

export default AdminPanel; 