// client/src/Login.js
import React, { useState } from 'react';
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from './firebaseConfig';
import { useAuth } from './context/AuthContext'; // To potentially redirect if already logged in
// Removed Link import as we use prop function now

// MUI Imports
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button'; // Standard Button
import LoadingButton from '@mui/lab/LoadingButton'; // Loading Button
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import Avatar from '@mui/material/Avatar';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Grid from '@mui/material/Grid'; // Import Grid for layout

// Accept the toggle function as a prop
function Login({ onSwitchToSignup }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { currentUser } = useAuth(); // Get current user state

  // Basic check to prevent showing login if already logged in
  // A router would handle this better with redirects.
  if (currentUser) {
      console.log("User already logged in, rendering null for Login component.");
      return null; // Don't render login form if user exists
  }

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    // Ensure auth was initialized
    if (!auth) {
        setError("Authentication service is unavailable. Please refresh or contact support.");
        setLoading(false);
        return;
    }
    try {
      await signInWithEmailAndPassword(auth, email, password);
      // Login successful - onAuthStateChanged in AuthContext handles setting the user state
      console.log("Login successful");
      // No navigation needed here without a router, AuthContext will trigger App re-render
    } catch (err) {
      // Use a more user-friendly error message if possible
      let friendlyError = 'Failed to log in. Please check email/password.';
      if (err.code === 'auth/user-not-found' || err.code === 'auth/wrong-password') {
         friendlyError = 'Invalid email or password.';
      } else if (err.message) {
         friendlyError = err.message; // Fallback to Firebase message
      }
      setError(friendlyError);
      console.error("Login error:", err.code, err.message);
    } finally {
        setLoading(false);
    }
  };

  return (
    <Container component="main" maxWidth="xs" sx={{ display: 'flex', alignItems: 'center', minHeight: '100vh' }}>
      <Paper 
        elevation={3} // Add some shadow
        sx={{
          marginTop: 8, // Keep margin from top
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          p: { xs: 2, sm: 3 }, // Responsive padding
          width: '100%', // Ensure Paper takes width within container
        }}
      >
        {/* Optional: Add an Avatar/Logo here later if desired */}
        {/* <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
          <LockOutlinedIcon />
        </Avatar> */}
        <Typography 
          component="h2" 
          variant="h4" /* Increased font size */
          color="primary" 
          sx={{
            mb: 1, 
            fontWeight: 'bold',
            // Add a subtle text shadow for effect
            textShadow: '1px 1px 3px rgba(0, 0, 0, 0.2)' 
          }} 
        >
          STUDIVA
        </Typography>
        <Typography component="h1" variant="h5">
          Sign In
        </Typography>
        {error && (
            <Alert severity="error" sx={{ width: '100%', mt: 2 }}>
                {error}
            </Alert>
        )}
        <Box component="form" onSubmit={handleLogin} noValidate sx={{ mt: 1, width: '100%' }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            error={!!error} // Highlight field if there was a login error
          />
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password"
            type="password"
            id="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            error={!!error} // Highlight field if there was a login error
          />
          {/* Use LoadingButton */}
          <LoadingButton
            type="submit"
            fullWidth
            variant="contained"
            loading={loading}
            sx={{ mt: 3, mb: 2 }}
          >
            Sign In
          </LoadingButton>
          
          {/* Use Button or Link-like Typography to trigger the switch */}
          <Grid container justifyContent="flex-end">
            <Grid item>
              <Button 
                variant="text" 
                onClick={onSwitchToSignup} // Call the passed function
                sx={{ textTransform: 'none' }} // Keep default text case
              >
                Don't have an account? Sign Up
              </Button>
            </Grid>
          </Grid>
          
        </Box>
      </Paper> { /* Close Paper instead of Box */ }
    </Container>
  );
}

export default Login; 