using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;

public class SenceSelector : MonoBehaviour
{
    public void XButton(InputAction.CallbackContext value)
    {
        float input = value.ReadValue<float>();
        if (input == 1)
        {
            Debug.Log("X was triggered.");
            SceneManager.LoadScene("HQ");
        }
    }
    public void YButton(InputAction.CallbackContext value)
    {
        float input = value.ReadValue<float>();
        if (input == 1)
        {
            Debug.Log("Y was triggered.");
            SceneManager.LoadScene("AHQ");
        }
    }

    public void MenuButton(InputAction.CallbackContext value)
    {
        float input = value.ReadValue<float>();
        if (input == 1)
        {
            Debug.Log("Menu was triggered.");
            SceneManager.LoadScene("Start");
        }
    }
}
