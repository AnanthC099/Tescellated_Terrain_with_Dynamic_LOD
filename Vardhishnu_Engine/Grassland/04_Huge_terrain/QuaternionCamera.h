#ifndef QUATERNION_CAMERA_H
#define QUATERNION_CAMERA_H

#include "Quaternion.h"
#include "glm/glm.hpp"

// Camera movement modes
enum CameraMode
{
    CAMERA_MODE_FREE,       // Free-fly camera (6DOF)
    CAMERA_MODE_FPS,        // First-person shooter style (constrained pitch)
    CAMERA_MODE_ORBIT       // Orbit around a target point
};

// Full-featured Quaternion-based Camera class
class QuaternionCamera
{
private:
    glm::vec3 m_position;           // Camera position in world space
    Quaternion m_orientation;        // Camera orientation as quaternion

    glm::vec3 m_targetPosition;      // For orbit mode - the point to orbit around

    float m_moveSpeed;               // Movement speed
    float m_rotationSpeed;           // Rotation speed (radians per pixel/unit)
    float m_orbitDistance;           // Distance from target in orbit mode

    float m_pitch;                   // Current pitch angle (for FPS mode clamping)
    float m_yaw;                     // Current yaw angle
    float m_roll;                    // Current roll angle

    float m_nearPlane;               // Near clipping plane
    float m_farPlane;                // Far clipping plane
    float m_fov;                     // Field of view in radians
    float m_aspectRatio;             // Aspect ratio (width/height)

    CameraMode m_mode;               // Current camera mode

    bool m_constrainPitch;           // Whether to constrain pitch (prevent gimbal lock issues)
    float m_maxPitch;                // Maximum pitch angle (for FPS mode)

    // Cached matrices
    mutable glm::mat4 m_viewMatrix;
    mutable glm::mat4 m_projectionMatrix;
    mutable bool m_viewDirty;
    mutable bool m_projDirty;

    void updateViewMatrix() const
    {
        if (!m_viewDirty) return;

        // Get the camera's axes from the quaternion
        glm::vec3 forward = m_orientation.getForward();
        glm::vec3 up = m_orientation.getUp();
        glm::vec3 right = m_orientation.getRight();

        // Build view matrix manually
        // View matrix = inverse of camera transformation
        // For orthonormal rotation, inverse = transpose
        m_viewMatrix = glm::mat4(1.0f);

        m_viewMatrix[0][0] = right.x;
        m_viewMatrix[1][0] = right.y;
        m_viewMatrix[2][0] = right.z;

        m_viewMatrix[0][1] = up.x;
        m_viewMatrix[1][1] = up.y;
        m_viewMatrix[2][1] = up.z;

        m_viewMatrix[0][2] = -forward.x;
        m_viewMatrix[1][2] = -forward.y;
        m_viewMatrix[2][2] = -forward.z;

        // Translation component
        m_viewMatrix[3][0] = -glm::dot(right, m_position);
        m_viewMatrix[3][1] = -glm::dot(up, m_position);
        m_viewMatrix[3][2] = glm::dot(forward, m_position);
        m_viewMatrix[3][3] = 1.0f;

        m_viewDirty = false;
    }

    void updateProjectionMatrix() const
    {
        if (!m_projDirty) return;

        // Build perspective projection matrix manually
        float tanHalfFov = std::tan(m_fov * 0.5f);

        m_projectionMatrix = glm::mat4(0.0f);

        m_projectionMatrix[0][0] = 1.0f / (m_aspectRatio * tanHalfFov);
        m_projectionMatrix[1][1] = 1.0f / tanHalfFov;
        m_projectionMatrix[2][2] = -(m_farPlane + m_nearPlane) / (m_farPlane - m_nearPlane);
        m_projectionMatrix[2][3] = -1.0f;
        m_projectionMatrix[3][2] = -(2.0f * m_farPlane * m_nearPlane) / (m_farPlane - m_nearPlane);

        m_projDirty = false;
    }

public:
    // Constructor with default values
    QuaternionCamera()
        : m_position(0.0f, 30.0f, 150.0f)
        , m_orientation()
        , m_targetPosition(0.0f, 0.0f, 0.0f)
        , m_moveSpeed(5.0f)
        , m_rotationSpeed(0.005f)
        , m_orbitDistance(100.0f)
        , m_pitch(0.0f)
        , m_yaw(0.0f)
        , m_roll(0.0f)
        , m_nearPlane(0.1f)
        , m_farPlane(2000.0f)
        , m_fov(glm::radians(45.0f))
        , m_aspectRatio(800.0f / 600.0f)
        , m_mode(CAMERA_MODE_FPS)
        , m_constrainPitch(true)
        , m_maxPitch(glm::radians(89.0f))
        , m_viewDirty(true)
        , m_projDirty(true)
    {
    }

    // Constructor with position
    QuaternionCamera(const glm::vec3& position, float fov = glm::radians(45.0f), float aspectRatio = 800.0f / 600.0f)
        : m_position(position)
        , m_orientation()
        , m_targetPosition(0.0f, 0.0f, 0.0f)
        , m_moveSpeed(5.0f)
        , m_rotationSpeed(0.005f)
        , m_orbitDistance(100.0f)
        , m_pitch(0.0f)
        , m_yaw(0.0f)
        , m_roll(0.0f)
        , m_nearPlane(0.1f)
        , m_farPlane(2000.0f)
        , m_fov(fov)
        , m_aspectRatio(aspectRatio)
        , m_mode(CAMERA_MODE_FPS)
        , m_constrainPitch(true)
        , m_maxPitch(glm::radians(89.0f))
        , m_viewDirty(true)
        , m_projDirty(true)
    {
    }

    // Set camera mode
    void setMode(CameraMode mode)
    {
        m_mode = mode;
        m_viewDirty = true;
    }

    CameraMode getMode() const { return m_mode; }

    // Position setters/getters
    void setPosition(const glm::vec3& position)
    {
        m_position = position;
        m_viewDirty = true;
    }

    void setPosition(float x, float y, float z)
    {
        m_position = glm::vec3(x, y, z);
        m_viewDirty = true;
    }

    const glm::vec3& getPosition() const { return m_position; }

    // Orientation setters/getters
    void setOrientation(const Quaternion& orientation)
    {
        m_orientation = orientation.normalized();
        m_viewDirty = true;
    }

    const Quaternion& getOrientation() const { return m_orientation; }

    // Look at a specific point
    void lookAt(const glm::vec3& target, const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f))
    {
        glm::vec3 direction = glm::normalize(target - m_position);

        // Calculate right and corrected up vectors
        glm::vec3 right = glm::normalize(glm::cross(up, -direction));
        glm::vec3 correctedUp = glm::cross(-direction, right);

        // Build rotation matrix
        glm::mat3 rotMatrix;
        rotMatrix[0] = right;
        rotMatrix[1] = correctedUp;
        rotMatrix[2] = -direction;

        m_orientation = Quaternion::fromMatrix(rotMatrix);

        // Update Euler angles from quaternion
        glm::vec3 euler = m_orientation.toEulerAngles();
        m_pitch = euler.x;
        m_yaw = euler.y;
        m_roll = euler.z;

        m_viewDirty = true;
    }

    // Movement functions
    void moveForward(float amount)
    {
        glm::vec3 forward = m_orientation.getForward();

        if (m_mode == CAMERA_MODE_FPS)
        {
            // In FPS mode, don't move up/down when looking up/down
            forward.y = 0.0f;
            if (glm::length(forward) > 0.001f)
                forward = glm::normalize(forward);
        }

        m_position += forward * amount * m_moveSpeed;
        m_viewDirty = true;
    }

    void moveBackward(float amount)
    {
        moveForward(-amount);
    }

    void moveRight(float amount)
    {
        glm::vec3 right = m_orientation.getRight();

        if (m_mode == CAMERA_MODE_FPS)
        {
            right.y = 0.0f;
            if (glm::length(right) > 0.001f)
                right = glm::normalize(right);
        }

        m_position += right * amount * m_moveSpeed;
        m_viewDirty = true;
    }

    void moveLeft(float amount)
    {
        moveRight(-amount);
    }

    void moveUp(float amount)
    {
        glm::vec3 up;

        if (m_mode == CAMERA_MODE_FPS)
        {
            // In FPS mode, always move along world Y axis
            up = glm::vec3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            up = m_orientation.getUp();
        }

        m_position += up * amount * m_moveSpeed;
        m_viewDirty = true;
    }

    void moveDown(float amount)
    {
        moveUp(-amount);
    }

    // Rotation functions using quaternions
    void rotate(float pitchDelta, float yawDelta, float rollDelta = 0.0f)
    {
        m_pitch += pitchDelta * m_rotationSpeed;
        m_yaw += yawDelta * m_rotationSpeed;
        m_roll += rollDelta * m_rotationSpeed;

        // Constrain pitch in FPS mode to prevent looking too far up/down
        if (m_constrainPitch && m_mode == CAMERA_MODE_FPS)
        {
            if (m_pitch > m_maxPitch) m_pitch = m_maxPitch;
            if (m_pitch < -m_maxPitch) m_pitch = -m_maxPitch;
        }

        // Build orientation quaternion from Euler angles
        // Order: Yaw (Y) -> Pitch (X) -> Roll (Z)
        Quaternion qYaw = Quaternion::fromAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), m_yaw);
        Quaternion qPitch = Quaternion::fromAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), m_pitch);
        Quaternion qRoll = Quaternion::fromAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), m_roll);

        m_orientation = qYaw * qPitch * qRoll;
        m_orientation.normalize();

        m_viewDirty = true;
    }

    // Yaw rotation (around world Y axis)
    void yaw(float angle)
    {
        m_yaw += angle;

        Quaternion qYaw = Quaternion::fromAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), m_yaw);
        Quaternion qPitch = Quaternion::fromAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), m_pitch);
        Quaternion qRoll = Quaternion::fromAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), m_roll);

        m_orientation = qYaw * qPitch * qRoll;
        m_orientation.normalize();

        m_viewDirty = true;
    }

    // Pitch rotation (around local X axis)
    void pitch(float angle)
    {
        m_pitch += angle;

        if (m_constrainPitch && m_mode == CAMERA_MODE_FPS)
        {
            if (m_pitch > m_maxPitch) m_pitch = m_maxPitch;
            if (m_pitch < -m_maxPitch) m_pitch = -m_maxPitch;
        }

        Quaternion qYaw = Quaternion::fromAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), m_yaw);
        Quaternion qPitch = Quaternion::fromAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), m_pitch);
        Quaternion qRoll = Quaternion::fromAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), m_roll);

        m_orientation = qYaw * qPitch * qRoll;
        m_orientation.normalize();

        m_viewDirty = true;
    }

    // Roll rotation (around local Z axis)
    void roll(float angle)
    {
        m_roll += angle;

        Quaternion qYaw = Quaternion::fromAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), m_yaw);
        Quaternion qPitch = Quaternion::fromAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), m_pitch);
        Quaternion qRoll = Quaternion::fromAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), m_roll);

        m_orientation = qYaw * qPitch * qRoll;
        m_orientation.normalize();

        m_viewDirty = true;
    }

    // Free rotation using quaternion multiplication (for free-fly mode)
    void rotateAroundAxis(const glm::vec3& axis, float angle)
    {
        Quaternion rotation = Quaternion::fromAxisAngle(axis, angle);
        m_orientation = rotation * m_orientation;
        m_orientation.normalize();

        // Update Euler angles
        glm::vec3 euler = m_orientation.toEulerAngles();
        m_pitch = euler.x;
        m_yaw = euler.y;
        m_roll = euler.z;

        m_viewDirty = true;
    }

    // Rotate around local axes (useful for free-fly camera)
    void rotateLocalX(float angle)
    {
        glm::vec3 localX = m_orientation.getRight();
        rotateAroundAxis(localX, angle);
    }

    void rotateLocalY(float angle)
    {
        glm::vec3 localY = m_orientation.getUp();
        rotateAroundAxis(localY, angle);
    }

    void rotateLocalZ(float angle)
    {
        glm::vec3 localZ = m_orientation.getForward();
        rotateAroundAxis(localZ, angle);
    }

    // Orbit mode functions
    void setOrbitTarget(const glm::vec3& target)
    {
        m_targetPosition = target;
        m_orbitDistance = glm::length(m_position - target);
        m_viewDirty = true;
    }

    void setOrbitDistance(float distance)
    {
        m_orbitDistance = distance;
        if (m_mode == CAMERA_MODE_ORBIT)
        {
            updateOrbitPosition();
        }
    }

    void orbit(float horizontalAngle, float verticalAngle)
    {
        if (m_mode != CAMERA_MODE_ORBIT) return;

        m_yaw += horizontalAngle;
        m_pitch += verticalAngle;

        // Clamp pitch to prevent flipping
        if (m_pitch > glm::radians(89.0f)) m_pitch = glm::radians(89.0f);
        if (m_pitch < glm::radians(-89.0f)) m_pitch = glm::radians(-89.0f);

        updateOrbitPosition();
    }

    void zoom(float amount)
    {
        if (m_mode == CAMERA_MODE_ORBIT)
        {
            m_orbitDistance -= amount * m_moveSpeed;
            if (m_orbitDistance < 1.0f) m_orbitDistance = 1.0f;
            updateOrbitPosition();
        }
        else
        {
            // For other modes, move forward/backward
            moveForward(amount);
        }
    }

private:
    void updateOrbitPosition()
    {
        // Calculate position on sphere around target
        m_position.x = m_targetPosition.x + m_orbitDistance * std::cos(m_pitch) * std::sin(m_yaw);
        m_position.y = m_targetPosition.y + m_orbitDistance * std::sin(m_pitch);
        m_position.z = m_targetPosition.z + m_orbitDistance * std::cos(m_pitch) * std::cos(m_yaw);

        // Look at target
        lookAt(m_targetPosition);
    }

public:
    // Projection settings
    void setPerspective(float fov, float aspectRatio, float nearPlane, float farPlane)
    {
        m_fov = fov;
        m_aspectRatio = aspectRatio;
        m_nearPlane = nearPlane;
        m_farPlane = farPlane;
        m_projDirty = true;
    }

    void setFOV(float fov)
    {
        m_fov = fov;
        m_projDirty = true;
    }

    void setAspectRatio(float aspectRatio)
    {
        m_aspectRatio = aspectRatio;
        m_projDirty = true;
    }

    void setNearPlane(float nearPlane)
    {
        m_nearPlane = nearPlane;
        m_projDirty = true;
    }

    void setFarPlane(float farPlane)
    {
        m_farPlane = farPlane;
        m_projDirty = true;
    }

    float getFOV() const { return m_fov; }
    float getAspectRatio() const { return m_aspectRatio; }
    float getNearPlane() const { return m_nearPlane; }
    float getFarPlane() const { return m_farPlane; }

    // Speed settings
    void setMoveSpeed(float speed) { m_moveSpeed = speed; }
    void setRotationSpeed(float speed) { m_rotationSpeed = speed; }
    float getMoveSpeed() const { return m_moveSpeed; }
    float getRotationSpeed() const { return m_rotationSpeed; }

    // Constraint settings
    void setConstrainPitch(bool constrain) { m_constrainPitch = constrain; }
    void setMaxPitch(float maxPitch) { m_maxPitch = maxPitch; }

    // Get matrices
    const glm::mat4& getViewMatrix() const
    {
        updateViewMatrix();
        return m_viewMatrix;
    }

    const glm::mat4& getProjectionMatrix() const
    {
        updateProjectionMatrix();
        return m_projectionMatrix;
    }

    // Get Vulkan-adjusted projection matrix (Y-flip for Vulkan coordinate system)
    glm::mat4 getProjectionMatrixVulkan() const
    {
        updateProjectionMatrix();
        glm::mat4 vulkanProj = m_projectionMatrix;
        vulkanProj[1][1] *= -1.0f;  // Flip Y for Vulkan
        return vulkanProj;
    }

    // Get direction vectors
    glm::vec3 getForward() const { return m_orientation.getForward(); }
    glm::vec3 getRight() const { return m_orientation.getRight(); }
    glm::vec3 getUp() const { return m_orientation.getUp(); }

    // Get Euler angles
    float getPitch() const { return m_pitch; }
    float getYaw() const { return m_yaw; }
    float getRoll() const { return m_roll; }

    // Reset camera to initial state
    void reset()
    {
        m_position = glm::vec3(0.0f, 30.0f, 150.0f);
        m_orientation = Quaternion::identity();
        m_pitch = 0.0f;
        m_yaw = 0.0f;
        m_roll = 0.0f;
        m_viewDirty = true;
    }

    // Process mouse movement input (deltaX, deltaY in pixels)
    void processMouseMovement(float deltaX, float deltaY, bool constrainPitch = true)
    {
        float xoffset = deltaX * m_rotationSpeed;
        float yoffset = deltaY * m_rotationSpeed;

        m_yaw -= xoffset;
        m_pitch -= yoffset;

        if (constrainPitch && m_mode == CAMERA_MODE_FPS)
        {
            if (m_pitch > m_maxPitch) m_pitch = m_maxPitch;
            if (m_pitch < -m_maxPitch) m_pitch = -m_maxPitch;
        }

        // Rebuild orientation quaternion
        Quaternion qYaw = Quaternion::fromAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), m_yaw);
        Quaternion qPitch = Quaternion::fromAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), m_pitch);
        Quaternion qRoll = Quaternion::fromAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), m_roll);

        m_orientation = qYaw * qPitch * qRoll;
        m_orientation.normalize();

        m_viewDirty = true;
    }

    // Smooth interpolation to a target position and orientation
    void smoothLookAt(const glm::vec3& targetPos, const Quaternion& targetOrientation, float t)
    {
        m_position = glm::mix(m_position, targetPos, t);
        m_orientation = Quaternion::slerp(m_orientation, targetOrientation, t);
        m_viewDirty = true;
    }

    // Create a view matrix for a given position and direction (utility function)
    static glm::mat4 createViewMatrix(const glm::vec3& position, const glm::vec3& direction, const glm::vec3& up)
    {
        glm::vec3 forward = glm::normalize(direction);
        glm::vec3 right = glm::normalize(glm::cross(up, -forward));
        glm::vec3 correctedUp = glm::cross(-forward, right);

        glm::mat4 viewMatrix(1.0f);

        viewMatrix[0][0] = right.x;
        viewMatrix[1][0] = right.y;
        viewMatrix[2][0] = right.z;

        viewMatrix[0][1] = correctedUp.x;
        viewMatrix[1][1] = correctedUp.y;
        viewMatrix[2][1] = correctedUp.z;

        viewMatrix[0][2] = -forward.x;
        viewMatrix[1][2] = -forward.y;
        viewMatrix[2][2] = -forward.z;

        viewMatrix[3][0] = -glm::dot(right, position);
        viewMatrix[3][1] = -glm::dot(correctedUp, position);
        viewMatrix[3][2] = glm::dot(forward, position);
        viewMatrix[3][3] = 1.0f;

        return viewMatrix;
    }
};

#endif // QUATERNION_CAMERA_H
