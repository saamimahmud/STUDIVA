import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// --- IMPORTANT ---
// Replace these placeholders with your actual Firebase project configuration!
// Find them in: Firebase Console > Project Settings > General > Your Apps > Web app > SDK setup and configuration > Config
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
  measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID
};

// Initialize Firebase
let app;
let auth;

try {
  app = initializeApp(firebaseConfig);
  auth = getAuth(app); // Get auth instance
  console.log("Firebase initialized successfully.");
} catch (error) {
  console.error("CRITICAL: Failed to initialize Firebase:", error);
  // Handle initialization error appropriately, maybe show an error message to the user
  // Setting auth to null or a dummy object might prevent further crashes but won't allow login
  auth = null;
}


export { auth }; // Export auth instance (or null if init failed) 