using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class InventoryManagerUI : MonoBehaviour
{
    [Header("Gameplay")]

    List<GameObject> itemObjects;
    int maxHeight; //how much place does inventory takes vertically, needed for scrolling correctly

    [Header("Pooling")]
    [SerializeField] int initialPoolSize = 50;
    [SerializeField] int overflowAdditionalPoolObjects = 10;
    public int currentPoolSize { get; protected set; }

    [Header("References")]
    [SerializeField] GameObject inventoryUI;
    public GameObject inventoryZone;
    [SerializeField] GameObject itemObject;
    [SerializeField] InventoryManager inv;
    public GameObject tempDragParent;

    int _debugItemsPlaced;

    void Start()
    {
        itemObjects = new List<GameObject>();
        UpdateItemObjectPool(initialPoolSize);
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.I))
        {
            ActivateUI();
        }
    }

    void UpdateItemObjectPool(int additionalSize)
    {
        for (int i = 0; i < additionalSize; i++)
        {
            GameObject newObj = Instantiate(itemObject, inventoryZone.transform);
            newObj.GetComponent<ItemObject>().Init(this);
            newObj.SetActive(false);
            itemObjects.Add(newObj);
            currentPoolSize++;
        }
    }

    public void UpdateItemVisual()
    {
        List<ItemStack> _items = inv.GetItemList();

        while (_items.Count > currentPoolSize)
        {
            UpdateItemObjectPool(overflowAdditionalPoolObjects);
        }

        for (int i = 0; i < _items.Count; i++)
        {
            ItemObject io = itemObjects[i].GetComponent<ItemObject>();
            io.SetItem(_items[i]);
            io.UpdateObject();
            io.gameObject.SetActive(true);
        }
        maxHeight = inv.GetGridHeight();
        UpdateInventoryZoneHeight();
    }

    public bool ActivateUI()
    {
        inventoryUI.SetActive(!inventoryUI.activeSelf);
        if (inventoryUI.activeSelf)
        {
            UpdateItemVisual();
        }
        return inventoryUI.activeSelf;
    }

    void UpdateInventoryZoneHeight()
    {
        inventoryZone.GetComponent<RectTransform>().sizeDelta = new Vector2(512, (maxHeight + 1) * 64);
    }

    public bool CheckFreeSpace(Vector2Int gridPos, Vector2Int size)
    {
        return inv.CheckIfInvPositionIsFree(gridPos, size.x, size.y);
    }

    public void ClearSpace(Vector2Int gridPos, Vector2Int size)
    {
        inv.ClearOccupiedSpace(gridPos, size.x, size.y);
    }

    public void FillSpace(Vector2Int gridPos, Vector2Int size)
    {
        inv.FillOccupiedSpace(gridPos, size.x, size.y);
    }
}
