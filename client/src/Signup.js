import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// MUI Imports
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button'; // For the switch to login button
import LoadingButton from '@mui/lab/LoadingButton';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import Avatar from '@mui/material/Avatar';
import PersonAddOutlinedIcon from '@mui/icons-material/PersonAddOutlined'; // Icon for signup
import Grid from '@mui/material/Grid';

function Signup({ onSwitchToLogin }) {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        if (password !== confirmPassword) {
            return setError('Passwords do not match');
        }
        if (!email || !password) {
            return setError('Email and password are required');
        }
        // Consider adding password strength validation here

        setLoading(true);
        try {
            const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
            const response = await fetch(`${apiUrl}/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            alert('Signup successful! Redirecting to login...');
            navigate('/login'); 
        } catch (err) {
            setError(err.message || 'Failed to sign up. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container component="main" maxWidth="xs" sx={{ display: 'flex', alignItems: 'center', minHeight: '100vh' }}>
            <Paper 
                elevation={3}
                sx={{
                    marginTop: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    p: { xs: 2, sm: 3 },
                    width: '100%',
                }}
            >
                <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
                    <PersonAddOutlinedIcon />
                </Avatar>
                <Typography component="h1" variant="h5">
                    Sign Up
                </Typography>
                {error && (
                    <Alert severity="error" sx={{ width: '100%', mt: 2 }}>
                        {error}
                    </Alert>
                )}
                <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1, width: '100%' }}>
                    <TextField
                        margin="normal"
                        fullWidth
                        id="name"
                        label="Name (Optional)"
                        name="name"
                        autoComplete="name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        disabled={loading}
                    />
                    <TextField
                        margin="normal"
                        required
                        fullWidth
                        id="email"
                        label="Email Address"
                        name="email"
                        autoComplete="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        disabled={loading}
                        error={!!error && (error.toLowerCase().includes('email') || error.toLowerCase().includes('required'))}
                    />
                    <TextField
                        margin="normal"
                        required
                        fullWidth
                        name="password"
                        label="Password"
                        type="password"
                        id="password"
                        autoComplete="new-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={loading}
                        error={!!error && (error.toLowerCase().includes('password') || error.toLowerCase().includes('required'))}
                    />
                    <TextField
                        margin="normal"
                        required
                        fullWidth
                        name="confirmPassword"
                        label="Confirm Password"
                        type="password"
                        id="confirmPassword"
                        autoComplete="new-password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        disabled={loading}
                        error={!!error && error.toLowerCase().includes('match')}
                    />
                    <LoadingButton
                        type="submit"
                        fullWidth
                        variant="contained"
                        loading={loading}
                        className="auth-button" // Keep class if you have specific shared styles
                        sx={{ mt: 3, mb: 2 }}
                    >
                        Sign Up
                    </LoadingButton>
                    <Grid container justifyContent="flex-end">
                        <Grid item>
                            <Button 
                                variant="text" 
                                onClick={onSwitchToLogin} 
                                sx={{ textTransform: 'none' }}
                                disabled={loading}
                            >
                                Already have an account? Sign In
                            </Button>
                        </Grid>
                    </Grid>
                </Box>
            </Paper>
        </Container>
    );
}

export default Signup; 