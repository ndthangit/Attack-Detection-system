import './Login.css'
import {Link} from "react-router-dom";
import { FaFacebook } from "react-icons/fa";
import {FcGoogle} from "react-icons/fc";

const Login = () => {

    // const navigate = useNavigate();
    // const handleSignup = () => {
    //     navigate('/signup');
    // }
    return (
        <div className="container">
            <header>Login Form</header>
            <form>
                <div className="input-field">
                    <input type="text" required/>
                    <label>Email or Username</label>
                </div>
                <div className="input-field">
                    <input className="pswrd" type="password" required/>
                    {/*<span className="show">SHOW</span>*/}
                    <label>Password</label>

                </div>
                <div className="button">
                    <div className="inner"></div>
                    <button>LOGIN</button>
                </div>
            </form>
            <div className="auth">
                Or login with
            </div>
            <div className="links">
                <div className="facebook">
                    <i className="fab fa-facebook-square">
                        <FaFacebook/>
                        <span>Facebook</span></i>
                </div>
                <div className="google">
                    <i className="fab fa-google-plus-square">
                        <FcGoogle />
                        <span>Google</span></i>
                </div>
            </div>
            <div className="signup">
                Not a member? <Link to="/signup">Signup</Link>
            </div>
        </div>

    )
}
export default Login