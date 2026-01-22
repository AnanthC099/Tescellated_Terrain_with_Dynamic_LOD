#ifndef QUATERNION_H
#define QUATERNION_H

#include <cmath>
#include "glm/glm.hpp"

// Custom Quaternion class - does not use GLM's quaternion implementation
// Implements all quaternion math from scratch
class Quaternion
{
public:
    float w, x, y, z;

    // Constructors
    Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}

    Quaternion(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}

    // Create quaternion from axis-angle representation
    static Quaternion fromAxisAngle(const glm::vec3& axis, float angleRadians)
    {
        float halfAngle = angleRadians * 0.5f;
        float sinHalf = std::sin(halfAngle);
        float cosHalf = std::cos(halfAngle);

        glm::vec3 normalizedAxis = glm::normalize(axis);

        return Quaternion(
            cosHalf,
            normalizedAxis.x * sinHalf,
            normalizedAxis.y * sinHalf,
            normalizedAxis.z * sinHalf
        );
    }

    // Create quaternion from Euler angles (pitch, yaw, roll in radians)
    static Quaternion fromEulerAngles(float pitch, float yaw, float roll)
    {
        float cy = std::cos(yaw * 0.5f);
        float sy = std::sin(yaw * 0.5f);
        float cp = std::cos(pitch * 0.5f);
        float sp = std::sin(pitch * 0.5f);
        float cr = std::cos(roll * 0.5f);
        float sr = std::sin(roll * 0.5f);

        Quaternion q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;

        return q;
    }

    // Magnitude (length) of quaternion
    float magnitude() const
    {
        return std::sqrt(w * w + x * x + y * y + z * z);
    }

    // Normalize the quaternion
    Quaternion normalized() const
    {
        float mag = magnitude();
        if (mag > 0.0001f)
        {
            return Quaternion(w / mag, x / mag, y / mag, z / mag);
        }
        return Quaternion();
    }

    // Normalize in place
    void normalize()
    {
        float mag = magnitude();
        if (mag > 0.0001f)
        {
            w /= mag;
            x /= mag;
            y /= mag;
            z /= mag;
        }
    }

    // Conjugate of quaternion
    Quaternion conjugate() const
    {
        return Quaternion(w, -x, -y, -z);
    }

    // Inverse of quaternion
    Quaternion inverse() const
    {
        float magSq = w * w + x * x + y * y + z * z;
        if (magSq > 0.0001f)
        {
            float invMagSq = 1.0f / magSq;
            return Quaternion(w * invMagSq, -x * invMagSq, -y * invMagSq, -z * invMagSq);
        }
        return Quaternion();
    }

    // Quaternion multiplication
    Quaternion operator*(const Quaternion& other) const
    {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }

    // Quaternion multiplication assignment
    Quaternion& operator*=(const Quaternion& other)
    {
        *this = *this * other;
        return *this;
    }

    // Scalar multiplication
    Quaternion operator*(float scalar) const
    {
        return Quaternion(w * scalar, x * scalar, y * scalar, z * scalar);
    }

    // Addition
    Quaternion operator+(const Quaternion& other) const
    {
        return Quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
    }

    // Rotate a vector by this quaternion
    glm::vec3 rotateVector(const glm::vec3& v) const
    {
        // q * v * q^(-1)
        // Optimized version using formula: v' = v + 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v))
        glm::vec3 qVec(x, y, z);
        glm::vec3 t = 2.0f * glm::cross(qVec, v);
        return v + w * t + glm::cross(qVec, t);
    }

    // Convert quaternion to 4x4 rotation matrix
    glm::mat4 toMatrix() const
    {
        Quaternion q = this->normalized();

        float xx = q.x * q.x;
        float xy = q.x * q.y;
        float xz = q.x * q.z;
        float xw = q.x * q.w;

        float yy = q.y * q.y;
        float yz = q.y * q.z;
        float yw = q.y * q.w;

        float zz = q.z * q.z;
        float zw = q.z * q.w;

        glm::mat4 result(1.0f);

        result[0][0] = 1.0f - 2.0f * (yy + zz);
        result[0][1] = 2.0f * (xy + zw);
        result[0][2] = 2.0f * (xz - yw);
        result[0][3] = 0.0f;

        result[1][0] = 2.0f * (xy - zw);
        result[1][1] = 1.0f - 2.0f * (xx + zz);
        result[1][2] = 2.0f * (yz + xw);
        result[1][3] = 0.0f;

        result[2][0] = 2.0f * (xz + yw);
        result[2][1] = 2.0f * (yz - xw);
        result[2][2] = 1.0f - 2.0f * (xx + yy);
        result[2][3] = 0.0f;

        result[3][0] = 0.0f;
        result[3][1] = 0.0f;
        result[3][2] = 0.0f;
        result[3][3] = 1.0f;

        return result;
    }

    // Convert to 3x3 rotation matrix
    glm::mat3 toMatrix3() const
    {
        Quaternion q = this->normalized();

        float xx = q.x * q.x;
        float xy = q.x * q.y;
        float xz = q.x * q.z;
        float xw = q.x * q.w;

        float yy = q.y * q.y;
        float yz = q.y * q.z;
        float yw = q.y * q.w;

        float zz = q.z * q.z;
        float zw = q.z * q.w;

        glm::mat3 result(1.0f);

        result[0][0] = 1.0f - 2.0f * (yy + zz);
        result[0][1] = 2.0f * (xy + zw);
        result[0][2] = 2.0f * (xz - yw);

        result[1][0] = 2.0f * (xy - zw);
        result[1][1] = 1.0f - 2.0f * (xx + zz);
        result[1][2] = 2.0f * (yz + xw);

        result[2][0] = 2.0f * (xz + yw);
        result[2][1] = 2.0f * (yz - xw);
        result[2][2] = 1.0f - 2.0f * (xx + yy);

        return result;
    }

    // Spherical linear interpolation (SLERP)
    static Quaternion slerp(const Quaternion& q1, const Quaternion& q2, float t)
    {
        Quaternion qa = q1.normalized();
        Quaternion qb = q2.normalized();

        // Compute dot product
        float dot = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;

        // If negative dot, negate one quaternion to take shorter path
        if (dot < 0.0f)
        {
            qb = Quaternion(-qb.w, -qb.x, -qb.y, -qb.z);
            dot = -dot;
        }

        // If quaternions are very close, use linear interpolation
        if (dot > 0.9995f)
        {
            Quaternion result(
                qa.w + t * (qb.w - qa.w),
                qa.x + t * (qb.x - qa.x),
                qa.y + t * (qb.y - qa.y),
                qa.z + t * (qb.z - qa.z)
            );
            return result.normalized();
        }

        // Standard SLERP
        float theta0 = std::acos(dot);
        float theta = theta0 * t;
        float sinTheta = std::sin(theta);
        float sinTheta0 = std::sin(theta0);

        float s0 = std::cos(theta) - dot * sinTheta / sinTheta0;
        float s1 = sinTheta / sinTheta0;

        return Quaternion(
            s0 * qa.w + s1 * qb.w,
            s0 * qa.x + s1 * qb.x,
            s0 * qa.y + s1 * qb.y,
            s0 * qa.z + s1 * qb.z
        );
    }

    // Linear interpolation (LERP) - faster but less accurate than SLERP
    static Quaternion lerp(const Quaternion& q1, const Quaternion& q2, float t)
    {
        float dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        Quaternion q2Adjusted = q2;
        if (dot < 0.0f)
        {
            q2Adjusted = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z);
        }

        Quaternion result(
            q1.w + t * (q2Adjusted.w - q1.w),
            q1.x + t * (q2Adjusted.x - q1.x),
            q1.y + t * (q2Adjusted.y - q1.y),
            q1.z + t * (q2Adjusted.z - q1.z)
        );

        return result.normalized();
    }

    // Get the forward direction vector (negative Z in OpenGL convention)
    glm::vec3 getForward() const
    {
        return rotateVector(glm::vec3(0.0f, 0.0f, -1.0f));
    }

    // Get the right direction vector (positive X)
    glm::vec3 getRight() const
    {
        return rotateVector(glm::vec3(1.0f, 0.0f, 0.0f));
    }

    // Get the up direction vector (positive Y)
    glm::vec3 getUp() const
    {
        return rotateVector(glm::vec3(0.0f, 1.0f, 0.0f));
    }

    // Extract Euler angles (pitch, yaw, roll) from quaternion
    glm::vec3 toEulerAngles() const
    {
        glm::vec3 angles;

        // Roll (x-axis rotation)
        float sinr_cosp = 2.0f * (w * x + y * z);
        float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        angles.z = std::atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        float sinp = 2.0f * (w * y - z * x);
        if (std::abs(sinp) >= 1.0f)
            angles.x = std::copysign(3.14159265358979323846f / 2.0f, sinp);
        else
            angles.x = std::asin(sinp);

        // Yaw (z-axis rotation)
        float siny_cosp = 2.0f * (w * z + x * y);
        float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        angles.y = std::atan2(siny_cosp, cosy_cosp);

        return angles;
    }

    // Dot product of two quaternions
    static float dot(const Quaternion& q1, const Quaternion& q2)
    {
        return q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
    }

    // Create a look-at quaternion
    static Quaternion lookAt(const glm::vec3& direction, const glm::vec3& up)
    {
        glm::vec3 forward = glm::normalize(direction);
        glm::vec3 right = glm::normalize(glm::cross(up, forward));
        glm::vec3 correctedUp = glm::cross(forward, right);

        // Build rotation matrix and convert to quaternion
        glm::mat3 rotMatrix;
        rotMatrix[0] = right;
        rotMatrix[1] = correctedUp;
        rotMatrix[2] = forward;

        return fromMatrix(rotMatrix);
    }

    // Create quaternion from rotation matrix
    static Quaternion fromMatrix(const glm::mat3& m)
    {
        Quaternion q;
        float trace = m[0][0] + m[1][1] + m[2][2];

        if (trace > 0.0f)
        {
            float s = 0.5f / std::sqrt(trace + 1.0f);
            q.w = 0.25f / s;
            q.x = (m[1][2] - m[2][1]) * s;
            q.y = (m[2][0] - m[0][2]) * s;
            q.z = (m[0][1] - m[1][0]) * s;
        }
        else
        {
            if (m[0][0] > m[1][1] && m[0][0] > m[2][2])
            {
                float s = 2.0f * std::sqrt(1.0f + m[0][0] - m[1][1] - m[2][2]);
                q.w = (m[1][2] - m[2][1]) / s;
                q.x = 0.25f * s;
                q.y = (m[1][0] + m[0][1]) / s;
                q.z = (m[2][0] + m[0][2]) / s;
            }
            else if (m[1][1] > m[2][2])
            {
                float s = 2.0f * std::sqrt(1.0f + m[1][1] - m[0][0] - m[2][2]);
                q.w = (m[2][0] - m[0][2]) / s;
                q.x = (m[1][0] + m[0][1]) / s;
                q.y = 0.25f * s;
                q.z = (m[2][1] + m[1][2]) / s;
            }
            else
            {
                float s = 2.0f * std::sqrt(1.0f + m[2][2] - m[0][0] - m[1][1]);
                q.w = (m[0][1] - m[1][0]) / s;
                q.x = (m[2][0] + m[0][2]) / s;
                q.y = (m[2][1] + m[1][2]) / s;
                q.z = 0.25f * s;
            }
        }

        return q.normalized();
    }

    // Identity quaternion
    static Quaternion identity()
    {
        return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
    }
};

#endif // QUATERNION_H
