// client/src/context/AuthContext.js
import React, { createContext, useState, useEffect, useContext } from 'react';
import { onAuthStateChanged } from "firebase/auth";
import { auth } from '../firebaseConfig'; // Import auth from your config

const AuthContext = createContext();

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isTeacher, setIsTeacher] = useState(false); // Store teacher status

  useEffect(() => {
    // Ensure auth was initialized before subscribing
    if (!auth) {
      console.error("Auth service not available in AuthProvider.");
      setLoading(false);
      return; // Stop if auth failed to initialize
    }

    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      if (user) {
        try {
          // Get token to check for custom claims
          const idTokenResult = await user.getIdTokenResult();
          setIsTeacher(idTokenResult.claims.teacher === true);
        } catch (error) {
          console.error("Error getting user claims:", error);
          setIsTeacher(false); // Assume not teacher on error
        } finally {
             setLoading(false);
        }
      } else {
        setIsTeacher(false); // Not logged in, not a teacher
        setLoading(false);
      }
      // setLoading(false); // Moved inside if/else/finally
    });

    return unsubscribe; // Cleanup subscription on unmount
  }, []);

  const value = {
    currentUser,
    isTeacher,
    loading // Include loading state
  };

  // Render children only when not loading initial auth state
  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
} 