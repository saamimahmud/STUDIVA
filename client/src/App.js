import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ReactMediaRecorder } from "react-media-recorder";
import './App.css';
// Import the new component
import CommPracticePanel from './CommPracticePanel';
// Import Admin Panel
import AdminPanel from './AdminPanel';
import SavedSessionsViewer from './SavedSessionsViewer'; // Import the new component
import Login from './Login'; // Import Login component
import { useAuth } from './context/AuthContext'; // Import useAuth hook
import { auth } from './firebaseConfig'; // Import auth instance
import { signOut } from 'firebase/auth'; // Import signOut

// MUI Imports
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import TextField from '@mui/material/TextField';
import Grid from '@mui/material/Grid'; // For layout
import CircularProgress from '@mui/material/CircularProgress'; // Loading indicator
import Alert from '@mui/material/Alert'; // For errors
import MicIcon from '@mui/icons-material/Mic'; // Example Icon
import StopIcon from '@mui/icons-material/Stop';
import SendIcon from '@mui/icons-material/Send';
import LogoutIcon from '@mui/icons-material/Logout';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TranslateIcon from '@mui/icons-material/Translate';
import SaveIcon from '@mui/icons-material/Save';

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
  const [currentMode, setCurrentMode] = useState('viva'); // 'viva', 'communication', 'admin', 'history'
  const [studentName, setStudentName] = useState(''); // State for student name

  const { currentUser, isTeacher } = useAuth(); // Get user and teacher status from context

  // --- State for Viva Mode ---
  const [studentAnswer, setStudentAnswer] = useState('');
  const [audioURL, setAudioURL] = useState(null);
  // Note: feedback and confidence are now part of 'results'
  const [questionIndex, setQuestionIndex] = useState(0);
  const [selectedSubject, setSelectedSubject] = useState('');
  const [questionMode, setQuestionMode] = useState('');
  const [questions, setQuestions] = useState([]);
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(false); // Specific loading state
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);
  const [vivaCompleted, setVivaCompleted] = useState(false);
  const [availableSubjects, setAvailableSubjects] = useState([]); // State for subjects
  const [subjectsError, setSubjectsError] = useState(null); // Optional: Error state for subjects fetch
  const [isSubmittingAnswer, setIsSubmittingAnswer] = useState(false); // For submit button loading
  const [isTranscribing, setIsTranscribing] = useState(false); // Already have this

  // --- State for Communication Mode (Add as needed) ---
  // This state will likely live within CommPracticePanel itself now

  // --- useEffect to Fetch Available Subjects ---
  useEffect(() => {
    const fetchSubjects = async () => {
      setSubjectsError(null);
      try {
        const response = await fetch('http://127.0.0.1:5000/subjects');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAvailableSubjects(data.sort()); // Sort subjects alphabetically
      } catch (e) {
        console.error("Failed to load subjects:", e);
        setSubjectsError(`Failed to load subjects: ${e.message}`);
      }
    };
    fetchSubjects();
  }, []); // Empty dependency array means run once on mount

  // --- useEffect for Viva Mode (fetching questions) ---
  useEffect(() => {
    // Only fetch questions if in viva mode
    if (currentMode === 'viva' && selectedSubject && questionMode) {
      const fetchQuestions = async () => {
        setIsLoadingQuestions(true);
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
          setIsLoadingQuestions(false);
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
    setIsSubmittingAnswer(true);
    setError(null);
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
        bert_f1_score: data.bert_f1_score,
        llm_evaluation: data.llm_evaluation,
        llm_reasoning: data.llm_reasoning,
        llm_confidence: data.llm_confidence,
        keyword_score: data.keyword_score,
        concept_match_score: data.concept_match_score,
        fluency_score: data.fluency_score,
        overall_score: data.overall_score,
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
    } finally {
      setIsSubmittingAnswer(false);
    }
  };

  const handleSaveSession = async () => {
    if (results.length === 0) {
      alert("No results to save.");
      return;
    }
    if (!studentName.trim()) { // Basic validation for student name
        alert("Please enter a student name before saving.");
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
          studentName: studentName.trim(), // Include student name
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
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
      }
      alert("Session saved successfully! ID: " + data.sessionId);
    } catch (err) {
      alert("Error saving session: " + err.message);
      console.error("Error saving session:", err);
    }
  };

  const handleAudioUpload = async () => {
    if (!audioURL) return;
    setIsTranscribing(true);
    setError(null);
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
      const errorMsg = `Transcription failed: ${err.response?.data?.error || err.message}`;
      setError(errorMsg);
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleLogout = async () => {
      try {
          await signOut(auth);
          setCurrentMode('viva'); // Reset mode on logout
          console.log("Logged out successfully");
      } catch (error) {
          console.error("Failed to log out:", error);
          alert("Failed to log out.");
      }
  };

  // --- Render Logic ---
  const renderVivaPanel = () => (
    <Paper elevation={3} sx={{ p: 3, mt: 2 }}> {/* Main panel container */}
       {/* Student Name Input */}
       <Box mb={3}>
            <TextField 
                label="Student Name"
                id="student-name"
                value={studentName}
                onChange={e => setStudentName(e.target.value)}
                placeholder="Enter student name or ID"
                required 
                fullWidth
                variant="outlined"
                // disabled={isLoadingQuestions || vivaCompleted || questionIndex > 0 } // Disable after starting?
            />
        </Box>

       {/* Selectors */} 
       <Grid container spacing={2} mb={3}>
            <Grid item xs={12} sm={6}> 
                <FormControl fullWidth variant="outlined" required disabled={isLoadingQuestions || !!subjectsError}> 
                    <InputLabel id="subject-select-label">Subject</InputLabel>
                    <Select
                        labelId="subject-select-label"
                        id="subject-select"
                        value={selectedSubject}
                        label="Subject"
                        onChange={(e) => setSelectedSubject(e.target.value)}
                    >
                        <MenuItem value="" disabled>-- Select Subject --</MenuItem>
                        {availableSubjects.map(subj => (
                            <MenuItem key={subj} value={subj}>{subj}</MenuItem>
                        ))}
                    </Select>
                    {subjectsError && <Typography color="error" variant="caption">{subjectsError}</Typography>}
                </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
                 <FormControl fullWidth variant="outlined" required disabled={!selectedSubject || isLoadingQuestions}>
                    <InputLabel id="mode-select-label">Mode</InputLabel>
                    <Select
                        labelId="mode-select-label"
                        id="mode-select"
                        value={questionMode}
                        label="Mode"
                        onChange={(e) => setQuestionMode(e.target.value)}
                    >
                        <MenuItem value="" disabled>-- Select Mode --</MenuItem>
                        <MenuItem value="sequential">Sequential</MenuItem>
                        <MenuItem value="random">Random</MenuItem>
                    </Select>
                </FormControl>
            </Grid>
        </Grid>

      {/* Loading/Error/Start Message */} 
      <Box sx={{ display: 'flex', justifyContent: 'center', minHeight: '50px', alignItems: 'center', mb: 2 }}>
            {isLoadingQuestions && <CircularProgress size={24} />}
            {error && <Alert severity="error" sx={{ width: '100%' }}>{error}</Alert>}
            {!isLoadingQuestions && !error && (!selectedSubject || !questionMode) && (
                <Typography variant="body1">Please select subject and mode.</Typography>
            )}
            {!isLoadingQuestions && !error && questions.length === 0 && selectedSubject && questionMode && (
                 <Typography variant="body1">No questions found for {selectedSubject}.</Typography>
            )}
      </Box>
     
      {/* Viva Interface */}
      {!vivaCompleted && !isLoadingQuestions && !error && questions.length > 0 && questions[questionIndex] ? (
        <Box>
            <Typography variant="h6" gutterBottom>
                Question {questionIndex + 1} of {questions.length}:
            </Typography>
            <Typography variant="body1" paragraph sx={{ fontWeight: 'bold' }}>
                {questions[questionIndex].question}
            </Typography>

            <TextField
                label="Your Answer"
                multiline
                rows={4}
                fullWidth
                variant="outlined"
                value={studentAnswer}
                onChange={e => setStudentAnswer(e.target.value)}
                placeholder="Type or record your answer..."
                sx={{ mb: 2 }}
                disabled={isSubmittingAnswer || isTranscribing}
            />

            <ReactMediaRecorder
                audio
                onStop={(blobUrl) => setAudioURL(blobUrl)}
                render={({ status, startRecording, stopRecording }) => (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Button 
                            onClick={startRecording} 
                            disabled={status === "recording" || isSubmittingAnswer || isTranscribing}
                            variant="outlined"
                            startIcon={<MicIcon />}
                        >
                           Record
                        </Button>
                        <Button 
                            onClick={stopRecording} 
                            disabled={status !== "recording" || isSubmittingAnswer || isTranscribing} 
                            variant="outlined"
                            color="secondary"
                            startIcon={<StopIcon />}
                         >
                            Stop
                         </Button>
                        <Typography variant="caption" sx={{ ml: 1 }}>Status: {status}</Typography>
                    </Box>
                )}
            />

            {audioURL && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                     <audio controls src={audioURL}></audio>
                     <Button 
                        onClick={handleAudioUpload} 
                        disabled={!audioURL || isSubmittingAnswer || isTranscribing}
                        variant="contained"
                        size="small"
                        startIcon={isTranscribing ? <CircularProgress size={16} color="inherit" /> : <TranslateIcon />}
                     >
                        {isTranscribing ? 'Transcribing...' : 'Use'}
                    </Button>
                </Box>
            )}

            <Button 
                onClick={handleSubmit} 
                disabled={!studentAnswer || isSubmittingAnswer || isTranscribing}
                variant="contained"
                color="primary"
                startIcon={isSubmittingAnswer ? <CircularProgress size={16} color="inherit" /> : <SendIcon />}
                fullWidth
            >
                Submit Answer
            </Button>
        </Box>
      ) : vivaCompleted ? (
        // Results Display - Refactored with MUI
        <Box mt={4}>
          <Typography variant="h5" gutterBottom>Viva Completed!</Typography>
          <TextField 
                label="Student Name (Read Only)" 
                value={studentName}
                fullWidth
                InputProps={{ readOnly: true }}
                variant="standard"
                sx={{ mb: 2 }}
            />
          <Typography variant="h6" gutterBottom>Results:</Typography>
          {results.map((result, index) => (
              <Paper key={index} elevation={1} sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom><strong>Q{index + 1}: {result.question}</strong></Typography>
                {/* Display Overall Score Prominently */}
                <Box sx={{ mb: 1.5, textAlign: 'center', p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                  <Typography variant="h6" component="span" sx={{ fontWeight: 'bold', color: (result.overall_score || 0) > 75 ? 'success.main' : (result.overall_score || 0) > 50 ? 'warning.main' : 'error.main' }}>
                    Overall Score: {(result.overall_score || 0).toFixed(2)}%
                  </Typography>
                </Box>
                {/* End Overall Score Display */}
                <Typography variant="body2" paragraph><strong>Your Answer:</strong> {result.answer}</Typography>
                <Typography variant="body2" paragraph><strong>Expected:</strong> {result.expected}</Typography>
                <Grid container spacing={1} sx={{ mb: 1 }}>
                    <Grid item xs={6} sm={3}><strong>Similarity:</strong> <span style={{ color: (result.similarity_score || 0) > 60 ? 'green' : 'red' }}>{(result.similarity_score || 0).toFixed(2)}%</span></Grid>
                    <Grid item xs={6} sm={3}><strong>BERT F1:</strong> <span style={{ color: (result.bert_f1_score || 0) > 70 ? 'green' : 'orange' }}>{(result.bert_f1_score || 0).toFixed(2)}%</span></Grid>
                    <Grid item xs={6} sm={3}><strong>Keywords:</strong> <span style={{ color: (result.keyword_score || 0) > 60 ? 'green' : 'red' }}>{(result.keyword_score || 0).toFixed(2)}%</span></Grid>
                    <Grid item xs={6} sm={3}><strong>Concepts:</strong> <span style={{ color: (result.concept_match_score || 0) > 60 ? 'green' : 'orange' }}>{(result.concept_match_score || 0).toFixed(2)}%</span></Grid>
                    <Grid item xs={6} sm={3}><strong>Fluency:</strong> {(result.fluency_score || 0).toFixed(2)}</Grid>
                </Grid>
                {/* LLM Evaluation Display */}
                {(result.llm_evaluation || result.llm_reasoning) && (
                    <Box sx={{ mt: 1.5, pt: 1.5, borderTop: 1, borderColor: 'divider' }}>
                        <Typography variant="body2" gutterBottom>
                            <strong>LLM Eval:</strong> {result.llm_evaluation || 'N/A'}
                        </Typography>
                        {result.llm_reasoning && (
                             <Typography variant="caption" display="block" sx={{ whiteSpace: 'pre-wrap', mt: 0.5, color: 'text.secondary' }}>
                                <strong>Reasoning:</strong> {result.llm_reasoning}
                             </Typography>
                        )}
                        {/* Optional: Display LLM confidence if available and meaningful */}
                        {/* <Typography variant="caption" display="block">LLM Confidence: {result.llm_confidence || 'N/A'}</Typography> */}
                    </Box>
                )}
                {/* End LLM Evaluation Display */}
                <Typography variant="body2" paragraph sx={{mt: 1}}><strong>Feedback:</strong> {result.feedback}</Typography>
                <Typography variant="body2"><strong>Confidence:</strong> 
                    <Box component="span" sx={{
                        bgcolor: result.confidence === "High" ? 'success.light' : result.confidence === "Medium" ? 'warning.light' : 'error.light',
                        color: result.confidence === "High" ? 'success.dark' : result.confidence === "Medium" ? 'warning.dark' : 'error.dark',
                        p: '2px 8px', borderRadius: '6px', ml: 1, fontWeight: 'bold' 
                    }}>
                        {result.confidence}
                    </Box>
                </Typography>
            </Paper>
          ))}
          <Button 
            onClick={handleSaveSession} 
            variant="contained"
            startIcon={<SaveIcon />} 
            sx={{ mt: 2 }}
            disabled={results.length === 0} // Disable if no results
          >
            Save Session
          </Button>
        </Box>
      ) : null}
    </Paper>
  );

  // If no user, force login view (basic implementation without routing)
  if (!currentUser) {
      return <Login />;
  }

  // If user is logged in, show the main app
  return (
    <Container maxWidth="lg" sx={{ mt: 2, mb: 4 }}> {/* Added mb: 4 for bottom spacing */}
          {/* AppBar for Header */}
          <AppBar position="static" elevation={1} sx={{ mb: 3 }}> {/* Added elevation=1 */}
            <Toolbar>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}> {/* Made title bold */} 
                    STUDIVA
                </Typography>
                <Typography variant="body2" sx={{ mr: 2 }}>
                    {currentUser.email} {isTeacher ? '(Teacher)' : '(Student)'}
                </Typography>
                <Button color="inherit" onClick={handleLogout} startIcon={<LogoutIcon />}>
                    Logout
                </Button>
            </Toolbar>
          </AppBar>

          {/* Mode Switching Buttons */} 
          <Box sx={{ mb: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Button variant={currentMode === 'viva' ? 'contained' : 'outlined'} onClick={() => setCurrentMode('viva')}>Viva Practice</Button>
                <Button variant={currentMode === 'communication' ? 'contained' : 'outlined'} onClick={() => setCurrentMode('communication')}>Communication Practice</Button>
                <Button variant={currentMode === 'history' ? 'contained' : 'outlined'} onClick={() => setCurrentMode('history')}>Session History</Button>
                {isTeacher && (
                    <Button variant={currentMode === 'admin' ? 'contained' : 'outlined'} onClick={() => setCurrentMode('admin')} color="secondary">
                        Admin Mode
                    </Button>
                )}
            </Box>

          {/* Main Content Area */}
          <Paper elevation={1} sx={{ p: { xs: 1, sm: 2, md: 3 } }}> {/* Add padding based on screen size */} 

                {/* === Hero Banner === */} 
                {(currentMode === 'viva' || (currentMode === 'admin' && isTeacher)) && (
                    <Box 
                        sx={{
                            textAlign: 'center',
                            mb: 3, // Margin bottom to separate from content below
                            p: 2, // Padding inside the banner
                            background: 'linear-gradient(45deg, #e3f2fd 30%, #bbdefb 90%)', // Example gradient (light blues)
                            borderRadius: '8px',
                            color: 'primary.dark' // Use a darker shade from the primary color
                        }}
                    >
                        <Typography variant="h5" component="p" sx={{ fontWeight: 'medium' }}>
                            "Empowering Smart Classrooms with AI"
                        </Typography>
                    </Box>
                )}
                {/* =================== */} 

                {currentMode === 'viva' && (
                    <div>
                        {renderVivaPanel()}
                    </div>
                )}
                {currentMode === 'communication' && (
                    <CommPracticePanel />
                )}
                {currentMode === 'history' && (
                    <SavedSessionsViewer />
                )}
                {currentMode === 'admin' && isTeacher && (
                    <AdminPanel />
                )}
                {currentMode === 'admin' && !isTeacher && (
                    <Alert severity="error">Access Denied. Admin mode is for teachers only.</Alert>
                )}
            </Paper>

            {/* === Footer === */} 
            <Box 
                component="footer" 
                sx={{
                    mt: 4, // Margin top to separate from content
                    py: 2, // Padding top and bottom
                    textAlign: 'center',
                    borderTop: '1px solid',
                    borderColor: 'divider', /
                    color: 'text.secondary' 
                }}
            >
                <Typography variant="body2">
                    Contact: Saami mahmud | saamimahmud123@gmail.com
                </Typography>
                 <Typography variant="caption" display="block" mt={1}>
                    © {new Date().getFullYear()} STUDIVA
                </Typography>
            </Box>
            {/* ============ */} 

    </Container>
  );
}

export default App;
