
import './App.css'

import {
    createBrowserRouter,
    RouterProvider
} from 'react-router-dom';
import Login from "./components/login/Login.jsx";
import Signup from "./components/signup/Signup.jsx";
import Upload from "./components/uploadNewData/upload.jsx";
import Navbar from "./components/narbar/Navbar.jsx";
import IndexConfig from "./components/index/IndexConfig.jsx";

function App() {
    const router = createBrowserRouter([
        {
            path: '/',
            element: <Navbar />
        },
        {
            path: '/upload',
            element: <Upload />
        },
        {
            path: '/template',
            element: <IndexConfig />
        },
        {
            path: '/login',
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
