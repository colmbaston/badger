use winit::
{
    dpi::PhysicalSize,
    event::{ KeyboardInput, ElementState, VirtualKeyCode }
};

const TAU : f32 = std::f32::consts::PI * 2.0;

pub struct Camera
{
    pub position           : glm::Vec3,
    pub forward_direction  : glm::Vec3,
    pub upward_direction   : glm::Vec3,
    pub fovx               : f32,
    pub near               : f32,
    pub far                : f32,
    pub yaw                : f32,
    pub pitch              : f32,
    pub mouse_sensitivity  : f32,
    pub scroll_sensitivity : f32,
    pub move_direction     : glm::Vec3,
    pub move_speed         : f32
}

impl Camera
{
    pub fn new() -> Camera
    {
        Camera
        {
            position:           glm::vec3(0.0, 0.0, 1.0),
            forward_direction:  glm::vec3(1.0, 0.0, 0.0),
            upward_direction:   glm::vec3(0.0, 0.0, 1.0),
            fovx:               TAU / 4.0,
            near:               0.1,
            far:                100.0,
            yaw:                0.0,
            pitch:              0.0,
            mouse_sensitivity:  0.01,
            scroll_sensitivity: 0.1,
            move_direction:     glm::vec3(0.0, 0.0, 0.0),
            move_speed:         3.0
        }
    }

    pub fn viewing_position(&self, orbit_radius : f32) -> glm::Vec3
    {
        self.position - orbit_radius * self.forward_direction
    }

    pub fn view(&self, orbit_radius : f32) -> glm::Mat4
    {
        glm::look_at_rh(&self.viewing_position(orbit_radius), &(self.position + self.forward_direction), &self.upward_direction)
    }

    pub fn projection(&self, PhysicalSize { width, height } : PhysicalSize<u32>) -> glm::Mat4
    {
        let aspect = width as f32 / height as f32;
        let fovy   = 2.0 * ((self.fovx / 2.0).tan() / aspect).atan();

        let mut matrix = glm::perspective_rh_zo(aspect, fovy, self.near, self.far);
        matrix[(1, 1)] *= -1.0;

        matrix
    }

    pub fn mouse_delta(&mut self, x : f32, y : f32)
    {
        self.yaw   = (self.yaw   + self.mouse_sensitivity * x) % TAU;
        self.pitch = (self.pitch - self.mouse_sensitivity * y).min(TAU / 4.001).max(-TAU / 4.001);
    }

    pub fn keyboard_input(&mut self, key : &KeyboardInput)
    {
        let movement_delta = match key.state
        {
            ElementState::Pressed  =>  1.0,
            ElementState::Released => -1.0
        };

        match key.virtual_keycode
        {
            Some(VirtualKeyCode::W) => self.move_direction.x += movement_delta,
            Some(VirtualKeyCode::S) => self.move_direction.x -= movement_delta,
            Some(VirtualKeyCode::A) => self.move_direction.y += movement_delta,
            Some(VirtualKeyCode::D) => self.move_direction.y -= movement_delta,
            Some(VirtualKeyCode::Q) => self.move_direction.z += movement_delta,
            Some(VirtualKeyCode::E) => self.move_direction.z -= movement_delta,
            _ => ()
        }
    }

    pub fn update(&mut self, frame_elapsed : f32)
    {
        self.forward_direction = glm::normalize(&glm::vec3(self.pitch.cos() *  self.yaw.cos(),
                                                           self.pitch.cos() * -self.yaw.sin(),
                                                           self.pitch.sin()));

        if self.move_direction.magnitude() > 0.0
        {
            let sideward_relative = glm::cross(&self.upward_direction, &self.forward_direction);
            let move_relative     = (self.move_direction.x * self.forward_direction)
                                  + (self.move_direction.y * sideward_relative)
                                  + (self.move_direction.z * glm::cross(&self.forward_direction, &sideward_relative));

            self.position += frame_elapsed * self.move_speed * glm::normalize(&move_relative);
        }
    }
}
