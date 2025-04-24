import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Define the custom theme using MUI's createTheme
const theme = createTheme({
  palette: {
    mode: 'light', // Explicitly set light mode
    primary: {
      main: '#007bff', // Use the blue from CSS variables
    },
    secondary: {
      main: '#6c757d', // Use the gray from CSS variables
    },
    background: {
      default: '#f8f9fa', // Light gray background
      paper: '#ffffff',   // White surface for components like Paper, Card
    },
    text: {
      primary: '#212529', // Dark gray for main text
      secondary: '#6c757d', // Lighter gray for secondary text
    },
    error: {
        main: '#dc3545', // Red for errors
    },
    // You can also add info, success, warning colors if needed
    // info: {
    //   main: '#17a2b8', // Teal from accent color
    // },
  },
  typography: {
    fontFamily: 'Inter, sans-serif', // Apply Inter font globally
    // You can further customize variants like h1, h2, body1, etc.
    // h1: {
    //   fontSize: '2.5rem',
    //   fontWeight: 700,
    // },
    // button: {
    //   textTransform: 'none', // Prevent default uppercase buttons if desired
    // }
  },
  // You can also customize components globally
  // components: {
  //   MuiButton: {
  //     styleOverrides: {
  //       root: {
  //         borderRadius: 8, // Example: Slightly rounder buttons
  //       },
  //     },
  //   },
  // }
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* Ensures baseline styles and background color are applied */}
      <AuthProvider>
        <App />
      </AuthProvider>
    </ThemeProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
