
import './App.css'

import {
    createBrowserRouter,
    RouterProvider
} from 'react-router-dom';
import Login from "./components/login/Login.jsx";
import Signup from "./components/signup/Signup.jsx";

function App() {
    const router = createBrowserRouter([
        {
            path: '/',
            element: <Login />
        },
        {
            path: '/signup',
            element: <Signup/>
        }

    ])

  return (
      <div>
          <RouterProvider router={router}/>
      </div>
  )
}

export default App
