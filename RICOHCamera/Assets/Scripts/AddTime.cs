using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using System.Linq;

public class AddTime : MonoBehaviour
{
    public GameObject TimeBall;
    private Vector3 pos_now;
    public RenderTexture ThetaV;
    public RenderTexture VideoWithTime;
    private RenderTexture timeText;
    public ComputeShader timeShader;
    private int kernelHandle;
    private long systemMs;

    // Start is called before the first frame update
    void Start()
    {
        timeText = new RenderTexture(3840, 1920, 0);
        timeText.enableRandomWrite = true;
        timeText.Create();

        kernelHandle = timeShader.FindKernel("Addtime");

    }

    // Update is called once per frame
    void Update()
    {
        Graphics.Blit(ThetaV, timeText);

        int[] timeCoords = timeCalculator();

        ComputeBuffer timeBuffer = new ComputeBuffer(timeCoords.Length, sizeof(int)); // Create a new ComputeBuffer
        timeBuffer.SetData(timeCoords);

        timeShader.SetBuffer(kernelHandle, "timeBuffer", timeBuffer);
        timeShader.SetTexture(kernelHandle, "Result", timeText);
        timeShader.Dispatch(kernelHandle, timeText.width / 8, timeText.height / 8, 1);

        timeBuffer.Release();
        Graphics.Blit(timeText, VideoWithTime);

    }

    int[] timeCalculator()
    {
        int time_hours = DateTime.Now.TimeOfDay.Hours;
        int time_minutes = DateTime.Now.TimeOfDay.Minutes;
        int time_seconds = DateTime.Now.TimeOfDay.Seconds;
        int time_milliseconds = DateTime.Now.TimeOfDay.Milliseconds;
        UnityEngine.Debug.Log(time_hours + ":" + time_minutes + ":" + time_seconds + ":" + time_milliseconds);

        // hours digit
        int firstNum = time_hours / 10;
        int secondNum = time_hours % 10;
        int[] hoursFirstDigit = numberCalculator(20, firstNum);
        int[] hoursSecondDigit = numberCalculator(80, secondNum);

        // minutes digit
        firstNum = time_minutes / 10;
        secondNum = time_minutes % 10;
        int[] minutesFirstDigit = numberCalculator(160, firstNum);
        int[] minutesSecondDigit = numberCalculator(220, secondNum);

        // seconds digit
        firstNum = time_seconds / 10;
        secondNum = time_seconds % 10;
        int[] secondsFirstDigit = numberCalculator(300, firstNum);
        int[] secondsSecondDigit = numberCalculator(360, secondNum);

        //milliseconds
        firstNum = time_milliseconds / 100;
        secondNum = time_milliseconds / 10 % 10;
        int thirdNum = time_seconds % 10;
        int[] millisecondsFirstDigit = numberCalculator(440, firstNum);
        int[] millisecondsSecondDigit = numberCalculator(500, secondNum);
        int[] millisecondsThirdDigit = numberCalculator(560, thirdNum);

        // time coordinates
        int[] timeCoords = hoursFirstDigit.Concat(hoursSecondDigit).Concat(minutesFirstDigit).Concat(minutesSecondDigit)
            .Concat(secondsFirstDigit).Concat(secondsSecondDigit).Concat(millisecondsFirstDigit).Concat(millisecondsSecondDigit)
            .Concat(millisecondsThirdDigit).ToArray();

        return timeCoords;
    }

    int[] numberCalculator(int startPixel, int num)
    {
        int[] digitNumber;
        int[] numCoords = new int[] { };
        switch (num)
        {
            case 0:
                // Code to execute if number is 0
                digitNumber = new int[] { 0, 1, 2, 4, 5, 6 };
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 1:
                // Code to execute if number is 1
                digitNumber = new int[] {2, 5};
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 2:
                // Code to execute if number is 2
                digitNumber = new int[] {0, 1, 3, 5, 6};
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 3:
                // Code to execute if number is 3
                digitNumber = new int[] {0, 2, 3, 5, 6};
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 4:
                // Code to execute if number is 4
                digitNumber = new int[] {2, 3, 4, 5};
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 5:
                // Code to execute if number is 5
                digitNumber = new int[] {0, 2, 3, 4, 6};
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 6:
                // Code to execute if number is 6
                digitNumber = new int[] {0, 1, 2, 3, 4, 6};
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 7:
                // Code to execute if number is 7
                digitNumber = new int[] {2, 5, 6};
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 8:
                // Code to execute if number is 8
                digitNumber = new int[] { 0, 1, 2, 3, 4, 5, 6 };
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
            case 9:
                // Code to execute if number is 9
                digitNumber = new int[] { 0, 2, 3, 4, 5, 6 };
                numCoords = new int[digitNumber.Length * 400];
                for (int i = 0; i < digitNumber.Length; i += 1)
                {
                    int[] tempCoords = digitCalculator(startPixel, digitNumber[i]);
                    Array.Copy(tempCoords, 0, numCoords, i * 400, tempCoords.Length);
                }
                break;
        }
        return numCoords;
    }

    int[] digitCalculator(int startPixel, int digit)
    {
        int[] digitCoords = new int[400];
        switch (digit) 
        {
            case 0:
                // Code to execute if digit is 0
                for (int i = 0; i < digitCoords.Length; i += 2)
                {
                    digitCoords[i] = startPixel + 5 + i / 2 % 40; // x coordinate
                    digitCoords[i + 1] = 10 + i / 2 / 40; // y coordinate
                }
                break;
            case 1:
                // Code to execute if digit is 1
                for (int i = 0; i < digitCoords.Length; i += 2)
                {
                    digitCoords[i] = startPixel + i / 2 % 5; // x coordinate
                    digitCoords[i + 1] = 10 + 5 + i / 2 / 5; // y coordinate
                }
                break;
            case 2:
                // Code to execute if digit is 2
                for (int i = 0; i < digitCoords.Length; i += 2)
                {
                    digitCoords[i] = startPixel + 45 + i / 2 % 5; // x coordinate
                    digitCoords[i + 1] = 10 + 5 + i / 2 / 5; // y coordinate
                }
                break;
            case 3:
                // Code to execute if digit is 3
                for (int i = 0; i < digitCoords.Length; i += 2)
                {
                    digitCoords[i] = startPixel + 5 + i /2 % 40; // x coordinate
                    digitCoords[i + 1] = 10 + 45 + i / 2 / 40; // y coordinate
                }
                break;
            case 4:
                // Code to execute if digit is 4
                for (int i = 0; i < digitCoords.Length; i += 2)
                {
                    digitCoords[i] = startPixel + i / 2 % 5; // x coordinate
                    digitCoords[i + 1] = 10 + 50 + i / 2 / 5; // y coordinate
                }
                break;
            case 5:
                // Code to execute if digit is 5
                for (int i = 0; i < digitCoords.Length; i += 2)
                {
                    digitCoords[i] = startPixel + 45 + i / 2 % 5; // x coordinate
                    digitCoords[i + 1] = 10 + 50 + i / 2 / 5; // y coordinate
                }
                break;
            case 6:
                // Code to execute if digit is 6
                for (int i = 0; i < digitCoords.Length; i += 2)
                {
                    digitCoords[i] = startPixel + 5 + i / 2 % 40; // x coordinate
                    digitCoords[i + 1] = 10 + 90 + i / 2 / 40; // y coordinate
                }
                break;
        }
        return digitCoords;
    }
}
