import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Import axios
import { ReactMediaRecorder } from "react-media-recorder"; // Import recorder
import { useAuth } from './context/AuthContext'; // Import useAuth

// MUI Imports
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import LoadingButton from '@mui/lab/LoadingButton'; // For loading states
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';

// MUI Icons (optional, could add later)
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import TranslateIcon from '@mui/icons-material/Translate';
import SendIcon from '@mui/icons-material/Send';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function CommPracticePanel() {
  // State for fetched scenarios
  const [scenariosData, setScenariosData] = useState([]);
  const [isLoadingScenarios, setIsLoadingScenarios] = useState(true); // Start loading true
  const [scenariosError, setScenariosError] = useState(null);
  
  // Keep selectedScenario storing the text string for now
  const [selectedScenario, setSelectedScenario] = useState(''); 
  const [response, setResponse] = useState('');
  const [commAudioURL, setCommAudioURL] = useState(null); // Add state for audio URL
  const [evaluation, setEvaluation] = useState(null);
  const [isLoadingEvaluation, setIsLoadingEvaluation] = useState(false); // Renamed isLoading
  const [isTranscribing, setIsTranscribing] = useState(false); // Add state for transcription loading
  const [error, setError] = useState(null);

  const { currentUser } = useAuth(); // Get current user

  // Fetch Scenarios on Mount
  useEffect(() => {
    const fetchScenarios = async () => {
      setIsLoadingScenarios(true);
      setScenariosError(null);

      if (!currentUser) {
        setScenariosError("User not logged in. Please log in to view scenarios.");
        setIsLoadingScenarios(false);
        setScenariosData([]); // Ensure data is empty
        return; // Don't attempt fetch if no user
      }

      let idToken;
      try {
        idToken = await currentUser.getIdToken();
      } catch (error) {
        console.error("Error getting ID token:", error);
        setScenariosError("Failed to get authentication token.");
        setIsLoadingScenarios(false);
        setScenariosData([]); // Ensure data is empty
        return;
      }

      try {
        // Fetch from the same endpoint used by AdminPanel, now with Auth header
        const response = await fetch(`${API_URL}/admin/scenarios`, {
            headers: {
                'Authorization': `Bearer ${idToken}`
            }
        }); 
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setScenariosData(data);
        // Set default selected scenario if data is available
        if (data.length > 0) {
             // Store only the text in selectedScenario state for now
            setSelectedScenario(data[0].scenario); 
        } else {
            setSelectedScenario(''); // No scenarios found
        }
      } catch (e) {
        // Handle potential 401/403 specifically if needed, otherwise generic message
        if (e.message.includes('401') || e.message.includes('403')) {
             setScenariosError(`Authentication failed. Please log in again.`);
        } else {
            setScenariosError(`Failed to load scenarios: ${e.message}`);
        }
        console.error("Failed to fetch scenarios:", e);
        setScenariosData([]); // Ensure data is empty on error
      } finally {
        setIsLoadingScenarios(false);
      }
    };

    fetchScenarios();
  }, [currentUser]); // Add currentUser to dependency array

  const handleSubmit = async () => {
    setIsLoadingEvaluation(true);
    setError(null);
    setEvaluation(null);

    try {
      const apiResponse = await fetch(`${API_URL}/evaluate_communication`, {
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
      setIsLoadingEvaluation(false);
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
      const response = await axios.post(`${API_URL}/transcribe`, formData, {
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
    <Box sx={{ p: { xs: 1, sm: 2 } }}> {/* Main container with padding */} 
      <Typography variant="h5" component="h2" gutterBottom>
          Classroom Communication Practice
      </Typography>
      <Typography variant="body1" paragraph color="text.secondary">
          Select a scenario and practice your response. Focus on politeness, clarity, and appropriateness.
      </Typography>

      {/* --- Scenario Selection --- */} 
      <Box sx={{ mb: 3 }}>
          {isLoadingScenarios && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={20} />
                  <Typography variant="body2" color="text.secondary">Loading scenarios...</Typography>
              </Box>
          )}
          {scenariosError && (
              <Alert severity="error" sx={{ mb: 1 }}>{scenariosError}</Alert>
          )}
          {!isLoadingScenarios && scenariosData.length === 0 && !scenariosError && (
             <Alert severity="info" sx={{ mb: 1 }}>No scenarios found. Please add scenarios in the Admin Panel.</Alert>
          )}
          {!isLoadingScenarios && scenariosData.length > 0 && (
              <FormControl fullWidth variant="outlined" disabled={isLoadingEvaluation || isTranscribing || isLoadingScenarios}>
                  <InputLabel id="scenario-select-label">Select Scenario</InputLabel>
                  <Select
                      labelId="scenario-select-label"
                      id="scenario-select"
                      value={selectedScenario} // Keep value as the scenario text string for now
                      label="Select Scenario"
                      onChange={e => {
                          setSelectedScenario(e.target.value);
                          setResponse('');
                          setCommAudioURL(null);
                          setEvaluation(null);
                          setError(null);
                      }}
                  >
                      {scenariosData.map((scenario) => (
                          <MenuItem key={scenario.id} value={scenario.scenario}>
                              {scenario.scenario}
                          </MenuItem>
                      ))}
                  </Select>
              </FormControl>
          )}
      </Box>
      {/* --- End Scenario Selection --- */} 

      {selectedScenario && (
          <Paper elevation={1} sx={{ p: { xs: 1, sm: 2 }, mt: 2 }}> {/* Wrap interaction area */} 
              {/* --- Response Input --- */} 
              <Box sx={{ mb: 2 }}>
                  <TextField
                      id="response-input"
                      label="Your Response (Type or Record)"
                      multiline
                      rows={4}
                      fullWidth
                      variant="outlined"
                      value={response}
                      onChange={e => setResponse(e.target.value)}
                      placeholder="Type your response here or use voice input below..."
                      disabled={isLoadingEvaluation || isTranscribing}
                  />
              </Box>
              {/* --- End Response Input --- */} 

              {/* --- Audio Recording --- */} 
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mb: 2, alignItems: 'center' }}>
                  <ReactMediaRecorder
                      audio
                      onStop={(blobUrl) => setCommAudioURL(blobUrl)}
                      render={({ status, startRecording, stopRecording }) => (
                          <Stack direction="row" spacing={1} alignItems="center">
                              <Button 
                                  onClick={startRecording} 
                                  disabled={status === "recording" || isLoadingEvaluation || isTranscribing}
                                  variant="outlined"
                                  startIcon={<MicIcon />}
                              >
                                  Record
                              </Button>
                              <Button 
                                  onClick={stopRecording} 
                                  disabled={status !== "recording" || isLoadingEvaluation || isTranscribing} 
                                  variant="outlined"
                                  color="secondary"
                                  startIcon={<StopIcon />}
                              >
                                  Stop
                              </Button>
                              <Typography variant="caption" sx={{ ml: 1 }}>Status: {status}</Typography>
                          </Stack>
                      )}
                  />
              </Stack>

              {commAudioURL && (
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mb: 2, alignItems: 'center' }}>
                      <audio controls src={commAudioURL} style={{ maxWidth: '100%' }}></audio> {/* Added maxWidth */} 
                      <LoadingButton 
                          onClick={handleCommAudioUpload} 
                          disabled={!commAudioURL || isLoadingEvaluation || isTranscribing}
                          loading={isTranscribing}
                          variant="contained"
                          size="small"
                          startIcon={<TranslateIcon />}
                          loadingPosition="start"
                      >
                          <span>Transcribe & Use</span>
                      </LoadingButton>
                  </Stack>
              )}
              {/* --- End Audio Recording --- */} 

              {/* --- Submit Button --- */} 
              <Box sx={{ mt: 2 }}>
                  <LoadingButton 
                      onClick={handleSubmit} 
                      disabled={isLoadingEvaluation || isTranscribing || !response || !selectedScenario}
                      loading={isLoadingEvaluation}
                      variant="contained"
                      color="primary"
                      fullWidth
                      startIcon={<SendIcon />}
                      loadingPosition="start"
                  >
                       <span>Submit for Evaluation</span>
                  </LoadingButton>
              </Box>
              {/* --- End Submit Button --- */} 
          </Paper>
      )}

      {/* --- Error Display --- */} 
      {error && <Alert severity="error" sx={{ mt: 2 }}>Error: {error}</Alert>}

      {/* --- Evaluation Results --- */} 
      {evaluation && (
        <Paper elevation={2} sx={{ mt: 3, p: { xs: 2, sm: 3 } }}>
            <Typography variant="h6" component="h3" gutterBottom>
                Evaluation Results:
            </Typography>
            <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={4}>
                    <Typography><strong>Tone:</strong> {evaluation.tone || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Typography><strong>Politeness:</strong> {evaluation.politeness}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Typography><strong>Clarity:</strong> {evaluation.clarity}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Typography><strong>Grammar:</strong> {evaluation.grammar}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Typography><strong>Appropriateness:</strong> {evaluation.appropriateness}</Typography>
                </Grid>
            </Grid>
            <Box mt={2}>
                 <Typography variant="subtitle1" gutterBottom><strong>Feedback & Suggestions:</strong></Typography>
                 {/* Use Box with pre-wrap instead of <pre> for better theme integration */}
                 <Box sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.9rem', background: (theme) => theme.palette.grey[100], p: 1.5, borderRadius: 1 }}>
                    {evaluation.feedback}
                 </Box>
            </Box>
        </Paper>
      )}
      {/* --- End Evaluation Results --- */} 
    </Box>
  );
}

export default CommPracticePanel; 