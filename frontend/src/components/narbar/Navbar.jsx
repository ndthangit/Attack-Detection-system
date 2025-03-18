import {Link} from "react-router-dom";
import './Navbar.css'
function Navbar() {

    return (
        <div className="navbar">
            <div className="left">
                <Link className='button font' to={'/'}>
                    <i className="fa-solid fa-house customed-icon"></i>
                    <div>Trang chủ</div>
                </Link>
                <Link className='button font' to={'/index'}>
                    <i className="fa-solid fa-calendar-days customed-icon"></i>
                    Index
                </Link>
                <Link className='button font' to={'/template'}>
                    <i className="fa-solid fa-clock-rotate-left customed-icon"></i>
                    Template
                </Link>
                <Link className='button font' to={'/upload'}>
                    <i className="fa-solid fa-dollar-sign customed-icon"></i>
                   Upload data
                </Link>
            </div>
        </div>

    )
}

export default Navbar