import { useState } from 'react';
import './upload.css';
import axios from 'axios';
import Navbar from "../narbar/Navbar.jsx";

const Upload = () => {
    const [file, setFile] = useState();
    const [uploadedFile, setUploadedFile] = useState();
    const [error, setError] = useState();


    function handleChange(event) {
        setFile(event.target.files[0]);
    }

    function handleSubmit(event) {
        event.preventDefault();
        const url = 'http://127.0.0.1:8000/upload/';
        const formData = new FormData();
        formData.append('file', file);
        formData.append('fileName', file.name);

        axios.post(url, formData)
            .then((response) => {
                console.log(response.data);
                setUploadedFile(response.data.file);
            })
            .catch((error) => {
                console.error("Error uploading file: ", error);
                setError(error);
            });
    }

    return (
        <div>
            <Navbar/>
            <div className="App">
                <form onSubmit={handleSubmit}>
                    <h1>React File Upload</h1>
                    <input type="file" onChange={handleChange}/>
                    <button type="submit">Upload</button>
                </form>
                {uploadedFile && <img src={uploadedFile} alt="Uploaded content"/>}
                {error && <p>Error uploading file: {error.message}</p>}

                <div className="field button-field">
                    <button>Save</button>
                </div>
            </div>
        </div>

    );
};

export default Upload;