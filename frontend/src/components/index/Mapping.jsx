import { useState } from "react";

// eslint-disable-next-line react/prop-types
const Mapping = ({ onChange }) => {
    // Sử dụng một state object để quản lý feature và dataType
    const [mapping, setMapping] = useState({
        feature: "",
        dataType: "keyword" // Giá trị mặc định
    });

    // Hàm xử lý thay đổi giá trị
    const handleChange = (e) => {
        const { name, value } = e.target;
        const updatedMapping = { ...mapping, [name]: value };
        setMapping(updatedMapping);

        // Gọi callback từ component cha (nếu có)
        if (onChange) {
            onChange(updatedMapping);
        }
    };

    return (
        <div className="mapping-content">
            <div className="field input-field">
                <label htmlFor="name-feature">Name feature</label>
                <input
                    type="text"
                    name="feature"
                    value={mapping.feature}
                    onChange={handleChange}
                    className="input"
                    // placeholder="Enter feature name"
                />
            </div>
            <div className="field select-field">
                <label htmlFor="data-type">Data type</label>
                <select
                    id="data-type"
                    name="dataType"
                    value={mapping.dataType}
                    onChange={handleChange}
                    className="input"
                >
                    <option value="keyword">Keyword</option>
                    <option value="string">String</option>
                    <option value="ip">Ip</option>
                    <option value="date">Date</option>
                    <option value="numeric">Numeric</option>
                    <option value="text">Text</option>
                    <option value="object">Object</option>
                </select>
            </div>
        </div>
    );
};

export default Mapping;