using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InventoryUI : MonoBehaviour
{
    [Header("Required Components")]
    public GameObject tempDragParent;
    [SerializeField] InventoryZoneUI[] controlledUIZones;

    void Start()
    {
        foreach (InventoryZoneUI izUI in controlledUIZones)
        {
            izUI.Init();
        }
    }
}
