import {useState} from "react";
import Mapping from "./Mapping.jsx";
import Navbar from "../narbar/Navbar.jsx";
import axios from "axios";

const IndexConfig =() => {
    const [settings, setSettings] = useState({
        number_of_shards: "3",
        number_of_replicas: "0",
        routing_allocation_include_tier_preference: "data_content"
    });
    const [mappings, setMappings] = useState([]);
    const [isDynamic, setIsDynamic] = useState(false);

    function handleCheckboxChange(event) {
        setIsDynamic(event.target.checked);
    }
    function handleChange(event) {
        const { name, value } = event.target;
        setSettings(prevSettings => ({
            ...prevSettings,
            [name]: value
        }));
    }

    const handleMappingChange = (index, updatedMapping) => {
        const updatedMappings = [...mappings];
        updatedMappings[index] = updatedMapping;
        setMappings(updatedMappings);
    };

    // Hàm thêm mapping mới
    const addMapping = () => {
        setMappings([...mappings, { feature: "", dataType: "keyword" }]);
    };

    /// Hàm xóa mapping cuối cùng
    const removeLastMapping = () => {
        if (mappings.length > 0) {
            const updatedMappings = mappings.slice(0, -1); // Xóa phần tử cuối cùng
            setMappings(updatedMappings);
        }
    };

    const handleSubmit = async () => {
        const template = {
            settings: {
                index: {
                    number_of_shards: settings.number_of_shards,
                    number_of_replicas: settings.number_of_replicas,
                    routing: {
                        allocation: {
                            include: {
                                _tier_preference: settings.routing_allocation_include_tier_preference
                            }
                        }
                    }
                }
            },
            mappings: {
                dynamic: isDynamic,
                properties: mappings.reduce((acc, mapping) => {
                    if (mapping.feature && mapping.dataType) {
                        acc[mapping.feature] = { type: mapping.dataType };
                    }
                    return acc;
                }, {})
            },
        };
        console.log(template);
        axios.post("http://127.0.0.1:8000/upload/create-indices", template, {
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then((response) => {
                console.log(response.data);
            })
            .catch((error) => {
                console.error("Error uploading file: ", error);
            });
    }

    return (
        <div>
            <Navbar/>
            <div className="template">
                <h1>Template</h1>

                <div>
                    <label>
                        <input type="checkbox" checked={isDynamic} onChange={handleCheckboxChange} />
                        Dynamic
                    </label>
                </div>
                <div className="settings-content">
                    <div className="field input-field">
                        <h2>Settings</h2>
                        <select id="settings" value={settings} onChange={handleChange} className="input">
                            <option value="standard">Standard</option>
                            <option value="time_series">Time Series</option>
                            <option value="logsdb">LogsDB</option>
                        </select>
                    </div>
                </div>
                <h1>Properties</h1>
                {mappings.map((mapping, index) => (
                    <div key={index}>
                        <Mapping
                            onChange={(updatedMapping) => handleMappingChange(index, updatedMapping)}
                        />
                    </div>
                ))}
                <div>
                    <button onClick={removeLastMapping} disabled={mappings.length === 1}>Remove Mapping</button>
                    <button onClick={addMapping}>Add Mapping</button>
                </div>

            </div>
            <div>
                <button onClick={handleSubmit}>Submit</button>
            </div>

        </div>


    )

}
export default IndexConfig;